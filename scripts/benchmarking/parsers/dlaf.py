import re

from .generic import RE_FLOAT, parse_cli

def parse_trsm(out_filepath):
    with open(out_filepath, 'r') as out_file:
        cli = out_file.readline()[:-1]

        nranks, cores_per_rank = parse_cli(cli)

        data = []
        for data_line in out_file.readlines():
            raw_data_line = data_line[:-1]
            m = re.match(f"\[(\d+)\] ({RE_FLOAT})s ({RE_FLOAT})GFlop/s \((\d+), (\d+)\) \((\d+), (\d+)\) \((\d+), (\d+)\) (\d+)", raw_data_line)
            if m:
                raw_data = m.groups()
                run_index, m, n, mb, nb, grid_rows, grid_cols, cores_per_task = [int(raw_data[n]) for n in [0, 3, 4, 5, 6, 7, 8, 9]]
                run_time, run_perf = [float(raw_data[n]) for n in [1, 2]]
                
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
            else:
                pass
        return (cli, data)

def parse_cholesky(out_filepath):
    with open(out_filepath, 'r') as out_file:
        cli = out_file.readline()[:-1]

        nranks, cores_per_rank = parse_cli(cli)

        data = []
        for data_line in out_file.readlines():
            raw_data_line = data_line[:-1]
            m = re.match(f"\[(\d+)\] ({RE_FLOAT})s ({RE_FLOAT})GFlop/s \((\d+), (\d+)\) \((\d+), (\d+)\) \((\d+), (\d+)\) (\d+)", raw_data_line)
            if m:
                raw_data = m.groups()
                run_index, m, n, mb, nb, grid_rows, grid_cols, cores_per_task = [int(raw_data[n]) for n in [0, 3, 4, 5, 6, 7, 8, 9]]
                run_time, run_perf = [float(raw_data[n]) for n in [1, 2]]
                
                data.append({
                    'run_index': run_index,
                    'm': m,
                    'mb': mb,
                    'grid_rows': grid_rows,
                    'grid_cols': grid_cols,
                    'time': run_time,
                    'performance': run_perf,
                })
            else:
                pass
        return (cli, data)

if __name__ == "__main__":
    for out_filepath in [
            'raw_data-cholesky/25658537J20200914_102004/25658537J20200914_102004.out',
            ]:
            data = parse_cholesky(out_filepath)
            for a in data:
                print(a)
    exit(1)

    for out_filepath in [
            'raw_data/25440303J20200904_160808/25440303J20200904_160808.out',
            'raw_data/25440303J20200904_161806/25440303J20200904_161806.out'
            ]:
            data = parse_trsm(out_filepath)
            for a in data:
                print(a)
