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
#ifndef CPP_CASS_EXTRACTOR_H
#define CPP_CASS_EXTRACTOR_H

#include "CassExtractorBase.h"

class CppCassExtractor : public CassExtractorBase {
public:
  CppCassExtractor(bool useCParser, bool allowParseErrors);
  ~CppCassExtractor();

protected:
  void collectFunctions(const TSNode &node) override;
  void collectFunctionsRec(const TSNode &node);

  std::unique_ptr<Cass> buildCass(const TSNode &node) override;
  std::unique_ptr<Cass> buildCassRec(const TSNode &node);

  void identifyLocalVariables(const TSNode &node) override;
  void identifyLocalVariablesRec(const TSNode &node, bool isDeclarator);
};

#endif // CPP_CASS_EXTRACTOR_H
