#include <fstream>
#include <sstream>

#include "CassExtractorBase.h"

CassExtractorBase::CassExtractorBase(bool allowParseErrors) {
  this->allowParseErrors = allowParseErrors;
  parser = ts_parser_new();
}

CassExtractorBase::~CassExtractorBase() { ts_parser_delete(parser); }

std::vector<std::pair<std::unique_ptr<Cass>, SourceRange>>
CassExtractorBase::extractCassForFunctions(const std::string &src,
                                           bool extractLoops) {
  srcBytes = src;

  TSTree *tree = ts_parser_parse_string(parser, nullptr, srcBytes.c_str(),
                                        srcBytes.length());

  auto root_node = ts_tree_root_node(tree);

  if (!allowParseErrors) {
    if (ts_node_has_error(root_node))
      return {};
  }

  collectedNodes.clear();
  if (extractLoops) {
    collectFunctionsAndLoops(root_node);
  } else {
    collectFunctions(root_node);
  }

  std::vector<std::pair<std::unique_ptr<Cass>, SourceRange>>
      cassesWithSrcRanges;
  for (const auto &node : collectedNodes) {
    symbolTable.clear();
    node2symbol.clear();
    identifyLocalVariables(node);

    node2cass.clear();
    auto cass = buildCass(node);
    if (cass) {
      auto start = ts_node_start_point(node);
      auto end = ts_node_end_point(node);
      cassesWithSrcRanges.emplace_back(
          std::move(cass),
          SourceRange{start.row, start.column, end.row, end.column});
    }
  }

  ts_tree_delete(tree);

  return cassesWithSrcRanges;
}
