import re

from .generic import RE_FLOAT, parse_cli

def parse_trsm(out_filepath):
    with open(out_filepath, 'r') as out_file:
        cli = out_file.readline()[:-1]

        nranks, cores_per_rank = parse_cli(cli)

        while True:
            data_line = out_file.readline()
            m = re.search('grid: (\d+) x (\d+)', data_line)
            if m:
                grid_rows, grid_cols = map(int, m.groups())
                break

        data = []
        flat_data = []
        run_index = 0
        for data_line in out_file.readlines():
            raw_data_line = data_line[:-1]

            if raw_data_line.startswith('size: '):
                m, n, nb = map(int, re.match('size: (\d+), (\d+); nb: (\d+)', raw_data_line).groups())
                flat_data.extend([m, n, nb])
            elif raw_data_line.startswith('scalapack elapsed'):
                run_time, run_perf = map(float, re.search(f'({RE_FLOAT}) s; ({RE_FLOAT}) Gflops', raw_data_line).groups())
                flat_data.extend([run_time, run_perf])
                            
            if len(flat_data) == 5:
                m, n, nb, run_time, run_perf = flat_data
                data.append({
                    'run_index': run_index,
                    'm': m,
                    'n': n,
                    'mb': nb,
                    'grid_rows': grid_rows,
                    'grid_cols': grid_cols,
                    'time': run_time,
                    'performance': run_perf,
                })

                flat_data = []
                run_index += 1
        return (cli, data)

def parse_cholesky(out_filepath):
    '''
    parse output of test_scalapack_real_chol
    '''
    with open(out_filepath, 'r') as out_file:
        cli = out_file.readline()[:-1]

        nranks, cores_per_rank = parse_cli(cli)

        while True:
            data_line = out_file.readline()
            m = re.search('grid: (\d+) x (\d+)', data_line)
            if m:
                grid_rows, grid_cols = map(int, m.groups())
                break

        data = []
        flat_data = []
        run_index = 0
        for data_line in out_file.readlines():
            raw_data_line = data_line[:-1]

            if raw_data_line.startswith('size: '):
                n, nb = map(int, re.match('size: (\d+); nb: (\d+)', raw_data_line).groups())
                flat_data.extend([n, nb])
            elif raw_data_line.startswith('scalapack elapsed'):
                run_time, run_perf = map(float, re.search(f'({RE_FLOAT}) s; ({RE_FLOAT}) Gflops', raw_data_line).groups())
                flat_data.extend([run_time, run_perf])
                            
            if len(flat_data) == 4:
                n, nb, run_time, run_perf = flat_data
                data.append({
                    'run_index': run_index,
                    'm': n,
                    'mb': nb,
                    'grid_rows': grid_rows,
                    'grid_cols': grid_cols,
                    'time': run_time,
                    'performance': run_perf,
                })

                flat_data = []
                run_index += 1
        return (cli, data)

if __name__ == "__main__":
    for out_filepath in [
            'raw_data-cholesky/25658520J20200914_101804/25658520J20200914_101804.out',
            ]:
            data = parse_cholesky(out_filepath)
            for a in data:
                print(a)

    for out_filepath in [
            'raw_data/25542683J20200908_110047/25542683J20200908_110047.out',
            'raw_data/25543208J20200908_114257/25543208J20200908_114257.out',
            ]:
            data = parse_trsm(out_filepath)
            for a in data:
                print(a)
