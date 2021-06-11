# Data Filtering

Filter out programs that cannot be parsed/compiled by tree-sitter/Clang.
```
python filter.py -c clang++ -d poj -i <POJ_dir> -o <POJ_out_dir> -f ./poj_filter_list.txt
python filter.py -c clang++ -d gcj -i <GCJ_dir> -o <GCJ_out_dir> -f ./gcj_filter_list.txt
```
