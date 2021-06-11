#ifndef CASS_EXTRACTOR_BASE_H
#define CASS_EXTRACTOR_BASE_H

#include <memory>
#include <string>
#include <tree_sitter/api.h>
#include <vector>

#include "Cass.h"
#include "SymbolTable.h"

struct SourceRange {
  uint32_t start_line;
  uint32_t start_column;
  uint32_t end_line;
  uint32_t end_column;
};

class CassExtractorBase {
public:
  CassExtractorBase(bool allowParseErrors);
  ~CassExtractorBase();

  std::vector<std::pair<std::unique_ptr<Cass>, SourceRange>>
  extractCassForFunctions(const std::string &src, bool extractLoops);

protected:
  bool allowParseErrors;
  TSParser *parser;
  std::string srcBytes;
  std::vector<TSNode> collectedNodes;
  SymbolTable symbolTable;
  std::unordered_map<const void *, Symbol *> node2symbol;
  std::unordered_map<const void *, Cass *> node2cass;

  virtual void collectFunctions(const TSNode &node) = 0;
  virtual void collectFunctionsAndLoops(const TSNode &node) = 0;

  virtual void identifyLocalVariables(const TSNode &node) = 0;

  virtual std::unique_ptr<Cass> buildCass(const TSNode &node) = 0;
};

#endif
