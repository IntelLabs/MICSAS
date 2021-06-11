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
