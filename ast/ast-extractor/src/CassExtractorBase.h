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
#ifndef CASS_EXTRACTOR_BASE_H
#define CASS_EXTRACTOR_BASE_H

#include <memory>
#include <string>
#include <tree_sitter/api.h>
#include <vector>

#include "Cass.h"
#include "SymbolTable.h"

class CassExtractorBase {
public:
  CassExtractorBase(bool allowParseErrors);
  ~CassExtractorBase();

  CassVec extractCassForFunctions(const std::string &src);

protected:
  bool allowParseErrors;
  TSParser *parser;
  std::string srcBytes;
  std::vector<TSNode> functions;
  SymbolTable symbolTable;
  std::unordered_map<const void *, Symbol *> node2symbol;
  std::unordered_map<const void *, Cass *> node2cass;

  virtual void collectFunctions(const TSNode &node) = 0;

  virtual void identifyLocalVariables(const TSNode &node) = 0;

  virtual std::unique_ptr<Cass> buildCass(const TSNode &node) = 0;
};

#endif
