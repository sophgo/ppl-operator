#!/usr/bin/env python3
import os
import argparse
import shutil
from test_case import full_list
import logging
from os_system import _os_subprocess, _os_system
from ppl.runtime.config import get_chip_code, get_chip_name
from contextlib import contextmanager
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s -\n%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

SUCCESS = 0
FAILURE = 1


class BaseTestFiles:

    def __init__(self, top_dir, chips, file_list, mode, is_full, time_out):
        self.result_message = ""
        self.top_dir = top_dir
        self.file_list = file_list
        self.chips = chips
        self.test_failed_list = []
        self.vali_failed_list = []
        self.time_out_list = []
        self.dubious_pl_list = {}
        self.file_not_found_list = {}
        self.chip_index_map = {
            'bm1684x': 1,
            'bm1688': 2,
            'bm1690': 3,
            'sg2260e': 4,
            'sg2262': 5,
            'sg2262rv': 6,
            'mars3': 7,
            'bm1684xe': 8,
        }
        self.mode = mode
        self.is_full = is_full
        self.time_out = time_out
        self.time_cost_list = []

    def summarize(self):
        self.result_message = f"\n====================== " \
                              f" {self.mode} test summarize ======================\n"
        if self.file_not_found_list:
            self.result_message += "\n[WARNING]: File does not exist:\n"
            for fileName, _ in self.file_not_found_list.items():
                self.result_message += f"- {fileName}\n"

        if self.test_failed_list:
            self.result_message += "[FAILED]: These PL files failed in compilation:\n"
            for chip, fileName in self.test_failed_list:
                self.result_message += f"- {fileName} tested in PLATFORM: {chip}\n"
        else:
            self.result_message += "[SUCCESS]: All PL files passed compilation\n"
        if self.mode == "validation":
            if self.vali_failed_list:
                self.result_message += "[FAILED]: These PL files do not passed the correctness validation:\n"
                for chip, fileName in self.vali_failed_list:
                    self.result_message += f"- {fileName} validated in PLATFORM: {chip}\n"
            else:
                self.result_message += "[SUCCESS]: All correctness validation passed.\n"

            if self.dubious_pl_list:
                self.result_message += "\n[WARNING]: These PL files do not have correctness validation scripts:\n"
                for fileName, _ in self.dubious_pl_list.items():
                    self.result_message += f"- {fileName}\n"

        if self.time_out_list:
            self.result_message += "\n[WARNING]: These PL files run time out:\n"
            for chip, fileName in self.time_out_list:
                self.result_message += f"- {fileName} tested in PLATFORM: {chip}\n"

        if self.time_cost_list:
            self.result_message += "\n[INFO]: The time consume summary:\n"
            for chip, time_cost, local_time_list in self.time_cost_list:
                self.result_message += f"- run models for {chip}: {time_cost:.1f} seconds\n"
                for file, ret, output_info, local_time_cost in local_time_list:
                    res = 'SUCCESS' if (ret == 0) else 'FAILED'
                    self.result_message += f"---- test {file} ({res}) [{local_time_cost:.1f} seconds]\n"
                    if ret != 0:
                        self.result_message += f"{output_info}\n"

    def check_test_open(self, case):
        flag = 1 if self.is_full else 0
        if type(case) == list:
            return case[flag]
        else:
            return case

    def get_applicable_tests(self, chip):
        applicable_tests = {}
        if chip not in self.chip_index_map:
            return applicable_tests
        chip_index = self.chip_index_map[chip]

        for category, tests in self.file_list.items():
            applicable_tests[category] = [
                test[0] for test in tests
                if self.check_test_open(test[chip_index])
            ]

        return applicable_tests

    def check_file_exists(self, path):
        if not os.path.exists(path):
            self.file_not_found_list[path] = ""
            logging.warning(f"[WARNING]: File does not exist - {path}")
            return False
        else:
            return True

    @contextmanager
    def work_dir(self):
        original_dir = os.getcwd()
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir)
        os.chdir(self.save_dir)
        try:
            yield
        finally:
            os.chdir(original_dir)

    def test_all(self):
        raise NotImplementedError("Subclasses should implement this method")


class TestPLFiles(BaseTestFiles):

    def __init__(self, top_dir, chips, file_list, save_dir, mode, is_full, time_out):
        super().__init__(top_dir, chips, file_list, mode, is_full, time_out)
        self.save_dir = save_dir
        self.is_full = is_full

    def test_one(self, fileName, chip):
        self.check_file_exists(fileName)
        logging.info(f"+++++++++++ testing {chip} {self.mode} {fileName} +++++++++++")
        
        test_opt = ""
        if self.mode == "performance":
            test_opt += " --autotune --mode pcie --without_profile"
        else:
            test_opt += " --gen_ref --mode cmodel"
        if chip == "sg2260e":
            test_opt += " --opt O3"
        elif chip == "sg2262rv":
            test_opt += " --rv"
        cmd = ["ppl_compile.py", "--src", fileName, "--chip", chip, test_opt]
        ret, output_info = _os_subprocess(cmd, self.time_out)
        if ret != 0:
            logging.error(f"ppl_compile failed with return code {ret}")
            if ret == 1:
                self.test_failed_list.append((chip, fileName))
            if ret == 2:
                self.time_out_list.append((chip, fileName))
        return ret, output_info


    def verify_one(self, fileName, chip):
        if chip == "sg2262rv":
            chip = "sg2262"
        testFile = fileName.replace(".pl", ".py")
        if os.path.exists(testFile):
            cmd = ["python", testFile, "--chip", chip]
            ret, output_info = _os_subprocess(cmd, self.time_out)
            if ret != 0:
                logging.error(f"verify failed with return code {ret}")
                self.vali_failed_list.append((chip, fileName))
                return ret, output_info
        else:
            logging.warning("File does not exist at the specified path.")
            self.dubious_pl_list[fileName] = ""
        return 0, ""

    def pack_perf(self, chip):
        print(self.top_dir )
        shutil.copy(os.path.join(os.environ["PPL_RUNTIME_PATH"],
                    f"chip/{get_chip_code(chip)}/lib/libtpudnn.so"), 
                    f"{self.save_dir}")
        shutil.copy(os.path.join(self.top_dir, f"ppl_perf/scripts/{chip}.sh"), 
                    f"{self.save_dir}/")
        cmd = ["tar", "-czf", f"./{chip}.tar.gz", f"{self.save_dir}"]
        _os_subprocess(cmd)

    def test_all(self):
        for _chip in self.chips:
            chip = get_chip_name(_chip)
            with self.work_dir():
                local_time_cost_list = []
                st = time.time()
                applicable_tests = self.get_applicable_tests(chip)
                for sub_dir, files in applicable_tests.items():
                    for file in files:
                        pl_file = os.path.join(self.top_dir, sub_dir, file)
                        st_local = time.time()
                        ret, output_info = self.test_one(pl_file, chip)
                        if ret == 0:
                            if self.mode == "validation":
                                ret, output_info = self.verify_one(pl_file, chip)
                        local_time_cost_list.append((pl_file, ret, output_info, time.time()-st_local))
                self.time_cost_list.append((chip, time.time()-st, local_time_cost_list))
            if self.mode == "performance":
                self.pack_perf(chip)
        self.summarize()
        return FAILURE if (len(self.file_not_found_list) or len(
            self.vali_failed_list) or len(self.test_failed_list)) else SUCCESS
    
@staticmethod
def clean_up_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
    os.chdir(dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir",
                        type=str,
                        default="./perf_out",
                        help="dir to save the test result")
    parser.add_argument("--test_mode",
                        type=str.lower,
                        default="full",
                        choices=["basic", "full"],
                        help="chip platform name")
    parser.add_argument("--chip",
                        type=str.lower,
                        default="bm1690,bm1684x",
                        help="chip platform name")
    parser.add_argument("--time_out",
                        type = int,
                        default = 0,
                        help="timeout")
    args = parser.parse_args()
    root = os.path.abspath(os.path.join(__file__, "..", ".."))
    chips = args.chip.split(",")
    is_full = True if args.test_mode == "full" else False
    correctness = TestPLFiles(root, chips, full_list, args.save_dir,
                              "validation", is_full, args.time_out)
    perf = TestPLFiles(root, chips, full_list, args.save_dir,
                       "performance", is_full, args.time_out)
    exit_status = 0
    testers = [correctness, perf]
    result_message = ""
    for test_runner in testers:
        if not isinstance(test_runner, BaseTestFiles):
            continue
        exit_status = test_runner.test_all() or exit_status
        result_message += test_runner.result_message
    logging.critical(result_message)
    exit(exit_status)
