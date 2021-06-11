from pathlib import Path
import subprocess
from collections import namedtuple
from typing import Tuple, Union, List

from ..cass import cass


CASS = cass.CassTree
SourceRange = namedtuple('SourceRange', ['start_line', 'start_column', 'end_line', 'end_column'])


class CASSManager:
    def __init__(self, use_c_parser: bool = False) -> None:
        cass_extractor: str = str(
            Path(__file__).parent.parent /
            'cass-extractor' / 'build' / 'bin' / 'cass-extractor'
        )
        self.extract_command = [cass_extractor]
        if use_c_parser:
            self.extract_command.append('-c')

    def extract_cass_strs_from_src_file(self, file_name: str, extract_loops: bool) -> List[str]:
        """
        Extract CASSes from the given source file. Return a list of serialized CASS strings.
        If extract_loops is True, each function and loop will be extracted as a CASS, otherwise only functions will be extracted.
        """
        cmd = self.extract_command + ['-l'] if extract_loops \
            else self.extract_command
        output = subprocess.check_output(
            cmd + ['-f', file_name], encoding='utf-8')
        return output.splitlines()

    def extract_cass_strs_from_src_text(self, src_text: Union[str, bytes], extract_loops: bool) -> List[str]:
        """
        Extract CASSes from the given source text. Return a list of serialized CASS strings.
        If extract_loops is True, each function and loop will be extracted as a CASS, otherwise only functions will be extracted.
        """
        cmd = self.extract_command + ['-l'] if extract_loops \
            else self.extract_command
        output = subprocess.check_output(cmd, input=src_text, encoding='utf-8')
        return output.splitlines()

    @staticmethod
    def load_cass_from_str(cass_str: str) -> Tuple[CASS, SourceRange]:
        c, src_range = cass.deserialize(cass_str)
        assert c
        return c, SourceRange(*src_range)

    @staticmethod
    def load_casses_from_strs(cass_strs: List[str]) -> Tuple[List[CASS], List[SourceRange]]:
        casses = []
        src_ranges = []
        for cass_str in cass_strs:
            c, rng = CASSManager.load_cass_from_str(cass_str)
            casses.append(c)
            src_ranges.append(rng)
        return casses, src_ranges
