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
  bool extractLoops = false;
  while ((c = getopt(argc, argv, "f:cdl")) != -1) {
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
    case 'l':
      extractLoops = true;
      break;
    case '?':
      std::cerr << "Usage: cass-extractor [-f <source_file>] [-c] [-d]"
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

  auto casses = extractor.extractCassForFunctions(src, extractLoops);

  if (dumpTree) {
    for (const auto &cassWithSrcRange : casses) {
      cassWithSrcRange.first->dump();
      std::cerr << std::endl;
    }
  }

  for (const auto &cassWithSrcRange : casses) {
    const auto &srcRange = cassWithSrcRange.second;
    std::cout << srcRange.start_line << ',' << srcRange.start_column << ','
              << srcRange.end_line << ',' << srcRange.end_column << '\t';
    Cass::serialize(cassWithSrcRange.first.get(), std::cout);
    std::cout << std::endl;
  }

  return 0;
}
