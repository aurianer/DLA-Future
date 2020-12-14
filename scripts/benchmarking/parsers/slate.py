import re

from .generic import RE_FLOAT, parse_cli

def parse_trsm(out_filepath):
    with open(out_filepath, 'r') as out_file:
        cli = out_file.readline()[:-1]
        data = []

        nranks, cores_per_rank = parse_cli(cli)

        while True:
            try:
                current_line = next(out_file)
                if current_line.startswith('type'):
                    break
            except StopIteration:
                return (cli, data)

        headers = re.split('\s+', current_line[:-1])

        run_index = 0
        for data_line in out_file.readlines():
            raw_data = re.split('\s{2,}', data_line)

            if len(raw_data) != len(headers) + 1:
                continue
            
            data_dict = {k:v for k, v in zip(headers, raw_data[1:])}

            m, n, mb, grid_rows, grid_cols = map(int, [data_dict[k] for k in [
                'm', 'n', 'nb', 'p', 'q']])

            run_time, run_perf = map(float, [data_dict[k] for k in [
                'time(s)', 'gflops']])

            data.append({
                'run_index': run_index,
                'm': m,
                'n': n,
                'mb': mb,
                'grid_rows': grid_rows,
                'grid_cols': grid_cols,
                'time': run_time,
                'performance': run_perf,
            })

            run_index += 1
        return (cli, data)

def parse_cholesky(out_filepath):
    with open(out_filepath, 'r') as out_file:
        cli = out_file.readline()[:-1]
        data = []

        nranks, cores_per_rank = parse_cli(cli)

        while True:
            try:
                current_line = next(out_file)
                if current_line.startswith('type'):
                    break
            except StopIteration:
                return (cli, data)

        headers = re.split('\s+', current_line[:-1])

        run_index = 0
        for data_line in out_file.readlines():
            raw_data = re.split('\s{2,}', data_line)

            if len(raw_data) != len(headers) + 1:
                continue
            
            data_dict = {k:v for k, v in zip(headers, raw_data[1:])}

            m, mb, grid_rows, grid_cols = map(int, [data_dict[k] for k in [
                'n', 'nb', 'p', 'q']])

            run_time, run_perf = map(float, [data_dict[k] for k in [
                'time(s)', 'gflops']])

            data.append({
                'run_index': run_index,
                'm': m,
                'mb': mb,
                'grid_rows': grid_rows,
                'grid_cols': grid_cols,
                'time': run_time,
                'performance': run_perf,
            })

            run_index += 1
        return (cli, data)

if __name__ == "__main__":
    for out_filepath in [
            #'raw_data/25515414J20200907_102351/25515414J20200907_102351.out',
            'raw_data/25597950J20200911_135116/25597950J20200911_135116.out',
            'raw_data/25597949J20200911_134751/25597949J20200911_134751.out',
            ]:
            data = parse_trsm(out_filepath)
            for a in data:
                print(a)
