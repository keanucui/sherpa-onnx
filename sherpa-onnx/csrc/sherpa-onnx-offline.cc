// sherpa-onnx/csrc/sherpa-onnx-offline.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include <stdio.h>

#include <chrono>  // NOLINT
#include <string>
#include <future>
#include <vector>
#include <fstream>
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/parse-options.h"
#include "sherpa-onnx/csrc/wave-reader.h"

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

int process(sherpa_onnx::OfflineRecognizer* recognizer,
             std::vector<std::string> file_names) {
    for (int32_t i = 0; i < file_names.size(); ++i) {
      std::string wav_filename = file_names[i];
      int32_t sampling_rate = -1;
      bool is_ok = false;
      std::vector<float> samples =
          sherpa_onnx::ReadWave(wav_filename, &sampling_rate, &is_ok);
      if (!is_ok) {
        fprintf(stderr, "Failed to read %s\n", wav_filename.c_str());
        return -1;
      }
      auto s = recognizer->CreateStream();
      s->AcceptWaveform(sampling_rate, samples.data(), samples.size());
      recognizer->DecodeStream(s.get());
      // std::string text = s->GetResult().AsJsonString();
      // fprintf(stderr, "%s\n", text.c_str());
    }
    return 0;
}

int main(int32_t argc, char *argv[]) {
  const char *kUsageMessage = R"usage(
Usage:

(1) Transducer from icefall

  ./bin/sherpa-onnx-offline \
    --tokens=/path/to/tokens.txt \
    --encoder=/path/to/encoder.onnx \
    --decoder=/path/to/decoder.onnx \
    --joiner=/path/to/joiner.onnx \
    --num-threads=2 \
    --decoding-method=greedy_search \
    /path/to/foo.wav [bar.wav foobar.wav ...]


(2) Paraformer from FunASR

  ./bin/sherpa-onnx-offline \
    --tokens=/path/to/tokens.txt \
    --paraformer=/path/to/model.onnx \
    --num-threads=2 \
    --decoding-method=greedy_search \
    /path/to/foo.wav [bar.wav foobar.wav ...]

Note: It supports decoding multiple files in batches

Default value for num_threads is 2.
Valid values for decoding_method: greedy_search.
foo.wav should be of single channel, 16-bit PCM encoded wave file; its
sampling rate can be arbitrary and does not need to be 16kHz.

Please refer to
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
for a list of pre-trained models to download.
)usage";
  bool use_wav_scp = false;    // true to use wav.scp as input
  int32_t nj = 10;  // true to use feats.scp as input
  int32_t batch_size = 10;

  sherpa_onnx::ParseOptions po(kUsageMessage);
  sherpa_onnx::OfflineRecognizerConfig config;
  config.Register(&po);
  po.Register("use_wav_scp", &use_wav_scp,
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

  fprintf(stderr, "%s\n", config.ToString().c_str());

  if (!config.Validate()) {
    fprintf(stderr, "Errors in config!\n");
    return -1;
  }

  fprintf(stderr, "Creating recognizer ...\n");
  const auto begin = std::chrono::steady_clock::now();
  std::vector<std::unique_ptr<sherpa_onnx::OfflineStream>> ss;
  std::vector<sherpa_onnx::OfflineStream *> ss_pointers;

  std::string wav_list = "./performance.txt";
  fprintf(stderr, "wav_list file name: %s\n", wav_list.c_str());
  std::vector<std::string> file_names = Read(wav_list);
  float duration = 0;
  for (auto wav_filename:file_names) {
    int32_t sampling_rate = -1;
    bool is_ok = false;
    const std::vector<float> samples =
        sherpa_onnx::ReadWave(wav_filename, &sampling_rate, &is_ok);
    if (!is_ok) {
      fprintf(stderr, "Failed to read %s\n", wav_filename.c_str());
      return -1;
    }
    duration += samples.size() / static_cast<float>(sampling_rate);
  }
  sherpa_onnx::OfflineRecognizer recognizer(config);
  std::vector<std::vector<std::string>> chunks_files = split(file_names, nj);
  ResourceListener resource_listener(::getpid());
  if (use_wav_scp) {
    std::vector<std::string> keys;
    float elapsed_seconds_all = 0;
  
    for (auto wav_filename:file_names) {
      int32_t sampling_rate = -1;
      bool is_ok = false;
      std::vector<float> samples =
          sherpa_onnx::ReadWave(wav_filename, &sampling_rate, &is_ok);
      if (!is_ok) {
        fprintf(stderr, "Failed to read %s\n", wav_filename.c_str());
        return -1;
      }
      keys.push_back(wav_filename);
      auto s = recognizer.CreateStream();
      s->AcceptWaveform(sampling_rate, samples.data(), samples.size());
      ss.push_back(std::move(s));
      ss_pointers.push_back(ss.back().get());
      if (static_cast<int32_t>(keys.size()) >= batch_size) {
        auto begin = std::chrono::steady_clock::now();
        recognizer.DecodeStreams(ss_pointers.data(), ss_pointers.size());
        auto end = std::chrono::steady_clock::now();
        float elapsed_seconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
            .count() /
        1000.;
        elapsed_seconds_all += elapsed_seconds;
     
        // for(int32_t i = 0; i < ss_pointers.size(); ++i) {
        //   std::string text = ss[i]->GetResult().AsJsonString().c_str();
        //   fprintf(stderr, "result: %s\n", text);
        // }
        keys.clear();
        ss.clear();
        ss_pointers.clear();
      }
    }
     
    if (!keys.empty()) {
      auto begin = std::chrono::steady_clock::now();
      recognizer.DecodeStreams(ss_pointers.data(), ss_pointers.size());
      auto end = std::chrono::steady_clock::now();
      float elapsed_seconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
            .count() /
        1000.;
      elapsed_seconds_all += elapsed_seconds;
      // for(int32_t i = 1; i <= ss_pointers.size(); ++i) {
      //   std::string text = ss[i - 1]->GetResult().AsJsonString().c_str();
      //   fprintf(stderr, "result: %s\n", text);
      // }
      keys.clear();
      ss.clear();
      ss_pointers.clear();
    }
    fprintf(stderr, "Elapsed seconds: %.3f s\n", elapsed_seconds_all);
    float rtf = duration / elapsed_seconds_all;
    fprintf(stderr, "Real time factor (RTF): %.3f / %.3f = %.3f\n",
            duration, elapsed_seconds_all, rtf);
    
  } else {
    std::vector<std::future<int>> fs;
    
    for (auto chunks:chunks_files) {
      fs.emplace_back(std::async(std::launch::async, process, &recognizer, chunks));
    }
    auto begin = std::chrono::steady_clock::now();
    for (const auto &each_f : fs) {
        each_f.wait();
    }
    auto end = std::chrono::steady_clock::now();
    resource_listener.ExitListen();
    float elapsed_seconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
            .count() /
        1000.;
    fprintf(stderr, "num threads: %d\n", config.model_config.num_threads);
    fprintf(stderr, "decoding method: %s\n", config.decoding_method.c_str());
    if (config.decoding_method == "modified_beam_search") {
      fprintf(stderr, "max active paths: %d\n", config.max_active_paths);
    }

    fprintf(stderr, "Elapsed seconds: %.3f s\n", elapsed_seconds);
    float rtf = duration / elapsed_seconds;
    fprintf(stderr, "Real time factor (RTF): %.3f / %.3f = %.3f\n",
            duration, elapsed_seconds, rtf);

  }

  return 0;
}
