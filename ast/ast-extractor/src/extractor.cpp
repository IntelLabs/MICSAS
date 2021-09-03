/*
MIT License

Copyright (c) 2021 Intel Labs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#include "CppCassExtractor.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <unistd.h>

int main(int argc, char *const *argv) {
  int c;
  char *filename = nullptr;
  bool dumpTree = false;
  bool useCParser = false;
  while ((c = getopt(argc, argv, "f:cd")) != -1) {
    switch (c) {
    case 'f':
      filename = optarg;
      break;
    case 'c':
      useCParser = true;
      break;
    case 'd':
      dumpTree = true;
      break;
    case '?':
      std::cerr << "Usage: ast-extractor [-f <source_file>] [-c] [-d]"
                << std::endl;
      return 1;
    }
  }

  std::string src;
  if (filename) {
    std::ifstream ifs(filename);
    std::stringstream buffer;
    buffer << ifs.rdbuf();
    src = buffer.str();
  } else {
    std::ostringstream buffer;
    buffer << std::cin.rdbuf();
    src = buffer.str();
  }

  CppCassExtractor extractor(useCParser, false);
  auto casses = extractor.extractCassForFunctions(src);

  if (casses.size() == 1 && casses[0] == nullptr) {
    return -1;
  }

  if (dumpTree) {
    for (const auto &cass : casses) {
      cass->dump();
      std::cerr << std::endl;
    }
  }

  for (const auto &cass : casses) {
    Cass::serialize(cass.get(), std::cout);
    std::cout << std::endl;
  }

  return 0;
}
