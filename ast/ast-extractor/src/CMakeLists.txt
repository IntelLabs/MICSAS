add_executable(cass-extractor
  extractor.cpp
  CassExtractorBase.cpp
  CppCassExtractor.cpp
  Cass.cpp
)

target_include_directories(cass-extractor
  PRIVATE
  ${TREE_SITTER_INCLUDE}
)

target_link_libraries(cass-extractor
  tree-sitter
  tree-sitter-c
  tree-sitter-cpp
)
