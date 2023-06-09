// sherpa-onnx/csrc/sherpa-onnx.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include <stdio.h>
#include <math.h>

#include <chrono>  // NOLINT
#include <string>
#include <vector>
#include <future>

#include "sherpa-onnx/csrc/online-recognizer.h"
#include "sherpa-onnx/csrc/online-stream.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/wave-reader.h"

#include <fstream>

#include "sherpa-onnx/csrc/cuidc_utils.h"

void Trim(std::string *str) {
  const char *white_chars = " \t\n\r\f\v";
/*delete white_chars in the front and end or delete the blank line.*/
  std::string::size_type pos = str->find_last_not_of(white_chars);
  if (pos != std::string::npos)  {
    str->erase(pos + 1);
    pos = str->find_first_not_of(white_chars);
    if (pos != std::string::npos) str->erase(0, pos);
  } else {
    str->erase(str->begin(), str->end());
  }
}

std::vector<std::string> Read(const std::string &rxfilename) {
  std::ifstream ifs(rxfilename.c_str(), std::ifstream::in);
  std::vector<std::string> lines;
  // there is no binary/non-binary mode.
  // char temp[UINT_MAX];
  if (ifs.fail()) {
      fprintf(stderr, "error: reading line ");
    return lines;  // probably eof.  fail in any case.
  }
  std::string cache;
  while (!ifs.eof()) {
    std::getline(ifs, cache);  // this will discard the \n, if present.
    Trim(&cache);
    if (!cache.empty()) {
      lines.push_back(cache);
    }
  }
  return lines;
}

std::vector<std::vector<std::string>> split(std::vector<std::string> input,
                                            size_t chunk_num) {
  std::vector<std::vector<std::string>> outputs;
  if (static_cast<unsigned>(input.size()) < chunk_num) {
    throw std::invalid_argument("输入的元素数量必须大于分块数.");
  }
  std::vector<unsigned> each_size(chunk_num, input.size() / chunk_num);
  auto last_num = input.size() % chunk_num;
  for (decltype(last_num) i = 0; i < last_num; ++i) {
    ++each_size[i];
  }

  auto itr = input.cbegin();
  for (const auto &size : each_size) {
    auto chunk_end = itr + size;
    outputs.emplace_back(itr, chunk_end);
    itr = chunk_end;
  }
  return outputs;
} 


int process(sherpa_onnx::OnlineRecognizerConfig  config,
             std::vector<std::string> file_names) {
  int32_t sampling_rate = -1;
  sherpa_onnx::OnlineRecognizer recognizer(config);
  for (auto wav_filename:file_names) {
    bool is_ok = false;
    std::vector<float> samples =
        sherpa_onnx::ReadWave(wav_filename, &sampling_rate, &is_ok);
    if (!is_ok) {
      fprintf(stderr, "Failed to read %s\n", wav_filename.c_str());
      return -1;
    }

    auto s = recognizer.CreateStream();
    s->AcceptWaveform(sampling_rate, samples.data(), samples.size());
    s->InputFinished();
    int frame_type = 0;
    while (recognizer.IsReady(s.get())) {
      recognizer.DecodeStream(s.get(), frame_type, wav_filename);
      frame_type += 1;
    }
  }
}

// TODO(fangjun): Use ParseOptions as we are getting more args
int main(int32_t argc, char *argv[]) {
  const char *usage = R"usage(
Usage:
  ./bin/sherpa-onnx \
    /path/to/tokens.txt \
    /path/to/encoder.onnx \
    /path/to/decoder.onnx \
    /path/to/joiner.onnx \
    /path/to/foo.wav [num_threads [decoding_method [/path/to/rnn_lm.onnx]]]

Default value for num_threads is 2.
Valid values for decoding_method: greedy_search (default), modified_beam_search.
foo.wav should be of single channel, 16-bit PCM encoded wave file; its
sampling rate can be arbitrary and does not need to be 16kHz.

Please refer to
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
for a list of pre-trained models to download.
)usage";
  bool single_process = false;    // true to use wav.scp as input
  int32_t nj = 1;  // true to use feats.scp as input
  int32_t batch_size = 10;

  sherpa_onnx::ParseOptions po(usage);
  sherpa_onnx::OnlineRecognizerConfig config;
  config.Register(&po);

  po.Register("single-process", &single_process,
              "If true, user should provide two arguments: "
              "scp:wav.scp ark,scp,t:results.ark,results.scp");
  po.Register("nj", &nj,
              "If true, user should provide two arguments: "
              "scp:wav.scp ark,scp,t:results.ark,results.scp");
  po.Register("batch-size", &batch_size,
              "Used only when --use-wav-scp=true or --use-feats-scp=true. "
              "It specifies the batch size to use for decoding");
  po.Read(argc, argv);

  if (po.NumArgs() < 1) {
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  if (!config.Validate()) {
    fprintf(stderr, "Errors in config!\n");
    return -1;
  }
  
  std::string wav_list = "./performance.txt";
  fprintf(stderr, "wav_list file name: %s\n", wav_list.c_str());
  std::vector<std::string> file_names = Read(wav_list);
  fprintf(stderr, "batch size %d\n", batch_size);
  int32_t sampling_rate = -1;
  float elapsed_seconds = 0;
  float duration = 0;
  
  std::vector<std::unique_ptr<sherpa_onnx::OnlineStream>> ss;
  int wavs_num = 0;
  for (auto wav_filename:file_names) {
    bool is_ok = false;
    std::vector<float> samples =
        sherpa_onnx::ReadWave(wav_filename, &sampling_rate, &is_ok);

    if (!is_ok) {
      fprintf(stderr, "Failed to read %s\n", wav_filename.c_str());
      return -1;
    }
    float duration_temp = samples.size() / static_cast<float>(sampling_rate);
    duration += duration_temp;
    wavs_num += 1;
  }
  int chunk_count = 0;
  ResourceListener resource_listener(::getpid());
  if (single_process) {
    sherpa_onnx::OnlineRecognizer recognizer(config);
    int chunk_num = ceil(wavs_num / float(batch_size));
    std::vector<std::vector<std::string>> chunks_files = split(file_names, chunk_num);
    for (auto file_names_one_chunk: chunks_files)  {
      std::string wav_name = "chunk_num_" + std::to_string(chunk_count);
      for (auto wav_filename:file_names_one_chunk) {
        bool is_ok = false;
        std::vector<float> samples =
            sherpa_onnx::ReadWave(wav_filename, &sampling_rate, &is_ok);

        if (!is_ok) {
          fprintf(stderr, "Failed to read %s\n", wav_filename.c_str());
          return -1;
        }
        auto begin = std::chrono::steady_clock::now();
        auto s = recognizer.CreateStream();
        s->AcceptWaveform(sampling_rate, samples.data(), samples.size());

        // std::vector<float> tail_paddings(static_cast<int>(0.5 * sampling_rate));
        // // Note: We can call AcceptWaveform() multiple times.
        // s->AcceptWaveform(sampling_rate, tail_paddings.data(), tail_paddings.size());

        // Call InputFinished() to indicate that no audio samples are available
        s->InputFinished();
        ss.push_back(std::move(s));
      }
      std::vector<sherpa_onnx::OnlineStream*> ss_pointers;
      int frame_type = 0;
      while (true) {
        for (size_t k=0; k < ss.size(); k++) {
          if (recognizer.IsReady(ss[k].get())) {
            ss_pointers.push_back(ss[k].get());
          }
        }
        if (ss_pointers.empty()) {
          ss.clear();
          ss_pointers.clear();
          break;
        }
        recognizer.DecodeStreams(ss_pointers.data(), ss_pointers.size(), frame_type, wav_name);
        frame_type += 1;
        ss_pointers.clear();
      }
      chunk_count += 1;
      // for (size_t k=0; k < ss.size(); k++) {
      //   std::string text = recognizer.GetResult(ss_pointers[k]).AsJsonString();
      //   fprintf(stderr, "Recognition result for: %s\n", text.c_str());
      // }
    }
  } else {
    std::vector<std::future<int>> fs;
    sherpa_onnx::OnlineRecognizer recognizer(config);
    std::vector<std::vector<std::string>> chunks_files = split(file_names, nj);
    for (auto chunks:chunks_files) {
      fs.emplace_back(std::async(process, config, chunks));
    }
    for (const auto &each_f : fs) {
        each_f.wait();
    }
  }
  resource_listener.ExitListen();
  fprintf(stderr, "num threads: %d\n", config.model_config.num_threads);
  fprintf(stderr, "decoding method: %s\n", config.decoding_method.c_str());
  if (config.decoding_method == "modified_beam_search") {
    fprintf(stderr, "max active paths: %d\n", config.max_active_paths);
  }

  fprintf(stderr, "Elapsed seconds: %.3f s\n", elapsed_seconds);
  float rtf = elapsed_seconds / duration;
  fprintf(stderr, "Real time factor (RTF): %.3f / %.3f = %.3f\n",
          elapsed_seconds, duration, rtf);

  return 0;
}
