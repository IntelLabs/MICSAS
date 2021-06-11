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
