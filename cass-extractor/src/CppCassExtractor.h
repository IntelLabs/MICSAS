#ifndef CPP_CASS_EXTRACTOR_H
#define CPP_CASS_EXTRACTOR_H

#include "CassExtractorBase.h"
#include <unordered_set>

class CppCassExtractor : public CassExtractorBase {
public:
  CppCassExtractor(bool useCParser, bool allowParseErrors);
  ~CppCassExtractor();

protected:
  static const std::unordered_set<std::string> function_node_types;
  static const std::unordered_set<std::string> loop_node_types;

  void collectFunctions(const TSNode &node) override;
  void collectFunctionsAndLoops(const TSNode &node) override;

  std::unique_ptr<Cass> buildCass(const TSNode &node) override;
  std::unique_ptr<Cass> buildCassRec(const TSNode &node);

  void identifyLocalVariables(const TSNode &node) override;
  void identifyLocalVariablesRec(const TSNode &node, bool isDeclarator);
};

#endif // CPP_CASS_EXTRACTOR_H
