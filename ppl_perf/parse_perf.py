import os, ast, glob, shutil, argparse
from os_system import _os_subprocess

def parse_line(line):
    list_str = line[line.find('['):len(line) - 1]
    strs = ast.literal_eval(list_str)
    return strs


def print_data(data, header):
    # print(tabulate(data, headers=header))
    head_len = []
    for x in range(len(header)):
        item_len = len(header[x])
        for item in data:
            item_len = max(len(str(item[x])), item_len)
        head_len.append(item_len)
    format_str = ["{:<" + str(x) + "}" for x in head_len]
    format_str = " ".join(format_str)
    print(format_str.format(*header))
    for item in data:
        print(format_str.format(*item))

def parse_libsophon_dma_data(line):
    data = line[line.find('[') + 1:line.find(']')]
    data = data.split(',')
    start_time = round(float(data[1]), 2)
    end_time = round(float(data[2]), 2)
    total_time = end_time - start_time
    speed_str = data[10]
    speed = float(speed_str[speed_str.find('=') + 1:speed_str.find('G')])
    data_size = speed * total_time * 1000
    return [start_time, end_time, total_time, speed, data_size]

def parse_libsophon_bd_data(line):
    data = line[line.find('[') + 1:line.find(']')]
    data = data.split(',')
    start_time = round(float(data[1]), 2)
    end_time = round(float(data[2]), 2)
    total_time = end_time - start_time
    return [start_time, end_time, total_time]





class PerfPaser:
    def __init__(self, path, chips, csv):
        self.path = path
        self.chips = chips
        self.csv = csv
        self.result = []
        
    @staticmethod
    def prepare_profile_tool():
        os.environ["PYTHONPATH"] = os.environ[
                "PPL_PROJECT_ROOT"] + "/third_party:" + os.environ["PYTHONPATH"]
        try:
            import bigTpuProfile
        except ImportError:
            cmd = ["pip", "install", "bigTpuProfile"]
            _os_subprocess(cmd)

    def parse_libsophon(self, path, name, chip):
        dma_data = []
        bd_data = []
        data_dir = os.path.join(path, "profile_data.js")
        if not os.path.exists(data_dir):
            raise Exception(f"parse error {data_dir} not exists")
        with open(data_dir) as file:
            for line in file:
                if line.find("gdma_id=") > 0 and line.find("speed") > 0:
                    dma_data.append(parse_libsophon_dma_data(line))
                if line.find("bd_id=") > 0 and line.find("cycle") > 0:
                    bd_data.append(parse_libsophon_bd_data(line))

        total_time = dma_data[-1][1] - dma_data[0][0]
        total_dma_time = sum([x[2] for x in dma_data])
        total_tiu_time = sum([x[2] for x in bd_data])
        total_data = sum([x[4] for x in dma_data])
        avg_bandwidth = total_data / total_dma_time / 1000
        para = (total_dma_time + total_tiu_time) / total_time
        performance_data = {
            'platform': chip,
            'name': name,
            'Parallelism': '%.2f%%' % (para * 100),
            'totalTime(us)': '%.2fus' % total_time,
            'TiuWorkingRatio': '%.2f%%' % (total_tiu_time / total_time * 100),
            'totalTiuTime(us)': '%.2fus' % total_tiu_time,
            'totalGdmaTime(us)': '%.2fus' % total_dma_time,
            'GdmaDdrAvgBandwidth(GB/s)': '%.2fGB/s' % avg_bandwidth,
        }
        self.result.append(performance_data)



    def parse_tpuv7(self, path, name, chip):
        data_dir = os.path.join(path, "PerfWeb/profile_data.js")
        if not os.path.exists(data_dir):
            raise Exception(f"parse error {data_dir} not exists")
        with open(data_dir) as file:
            header_item = []
            data_item = []
            for line in file:
                if line.find("summary_header") > 0:
                    header_item = parse_line(line)
                if line.find("summary_data") > 0:
                    data_item = parse_line(line)
            if len(header_item) == 0 or len(data_item) == 0:
                raise Exception("parse error")
            if len(header_item) < 9:
                raise Exception("parse error")
            data = data_item[-1]
            performance_data = {
                'platform': chip,
                'name': name,
                'Parallelism': data[1],
                'totalTime(us)': f'{data[2]}us',
                'TiuWorkingRatio': data[3],
                'totalTiuTime(us)': data[4],
                'uArchURate': data[5],
                'totalGdmaTime(us)': data[6],
                'GdmaDdrAvgBandwidth(GB/s)': f'{data[7]}GB/s',
                'GdmaL2AvgBandwidth(GB/s)': f'{data[8]}GB/s'
            }
            self.result.append(performance_data)

    def save_report(self):
        try: 
            import pandas as pd
        except ImportError:
            cmd = ["pip", "install", "pandas"]
            _os_subprocess(cmd)
            import pandas as pd
        df = pd.DataFrame(self.result)
        df.to_csv(self.csv, index=False)

    def run(self):
        self.prepare_profile_tool()
        original_dir = os.getcwd()
        os.chdir(self.path)
        for chip in self.chips:
            ret = 1
            tar_name = f"profiling_{chip}.tar.gz"
            if os.path.exists(tar_name):
                if os.path.exists(f"profiling_{chip}"):
                    shutil.rmtree(f"profiling_{chip}")
                cmd = ['tar', 'xzf', tar_name, './']
                ret, output_info = _os_subprocess(cmd)
                if ret != 0:
                    print(output_info)
                    print("[ERROR] tar extraction failed!")
                    return ret
                pattern = os.path.join(f"*profiling_{chip}*", "*", "bmprofile_data*")
                cmd = ["python -m bmprofile --mode time"]
                if chip == "bm1690":
                    pattern = os.path.join(f"*profiling_{chip}*", "*", "cdm_profile_data_dev*")
                    cmd = ["bigTpuProfile", "--disable_doc"]
                dirs = glob.glob(pattern, recursive=True)
                for item in dirs:
                    o_path = os.path.join(os.path.dirname(item), "perf_out")
                    case = item.split('/')[-2].replace("test_", "").split('-')[0]
                    _cmd = cmd + [item, o_path]
                    ret, output_info = _os_subprocess(_cmd)
                    if ret != 0:
                        print(output_info)
                        print("[ERROR] run profile tool failed!")
                        return ret
                    if chip == "bm1684x":
                        self.parse_libsophon(o_path, case, chip)
                    else:
                        self.parse_tpuv7(o_path, case, chip)
        os.chdir(original_dir)
        self.save_report()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",
                        type=str,
                        required=True,
                        help="file path to csv result")
    parser.add_argument("--dir",
                        type=str,
                        required=True,
                        help="dir to perf data")
    parser.add_argument("--chip",
                        type=str.lower,
                        # default="bm1684x,bm1688,bm1690,sg2262,mars3,bm1684xe,sg2260e,sg2262rv",
                        default="bm1690,bm1684x",
                        help="chip platform name")

    args = parser.parse_args()
    chips = args.chip.split(",")

    parser = PerfPaser(args.dir, chips, args.csv)
    parser.run()
