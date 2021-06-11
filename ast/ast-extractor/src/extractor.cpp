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
