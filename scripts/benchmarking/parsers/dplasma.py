import re

from .generic import RE_FLOAT, parse_cli

def parse_trsm(out_filepath):
    with open(out_filepath, 'r') as out_file:
        cli = out_file.readline()[:-1]

        nranks, cores_per_rank = parse_cli(cli)
        
        fields = [
            'run_index',
            'm',
            'n',
            'mb',
            'grid_rows',
            'grid_cols',
            'time',
            'performance',
            ]
        data = { k:v for k,v in zip(fields, len(fields)*[None]) }

        data['run_index'] = 1

        for data_line in out_file.readlines():
            raw_data_line = data_line[:-1]

            if raw_data_line.startswith(r'#+++++'):
                field, raw_value = raw_data_line[7:].split(':')
                if 'P x Q' in field:
                    data['grid_rows'], data['grid_cols'] = map(int, re.search(r'(\d+)\s+x\s+(\d+)\s*.*', raw_value).groups())
                elif 'M x N x K' in field:
                    data['m'], data['n'] = map(int, re.search(r'(\d+)\s+x\s+(\d+)\s*.*', raw_value).groups())
                elif 'MB x NB' in field:
                    data['mb'], data['nb'] = map(int, re.search(r'(\d+)\s+x\s+(\d+)', raw_value).groups())
            elif raw_data_line.startswith(r'[****] TIME(s)'):
                raw_data = raw_data_line.split(':')
                raw_time = raw_data[0]
                raw_perf = raw_data[-1]
                data['time'] = float(re.search(f'({RE_FLOAT})', raw_time).group(1))
                data['performance'] = float(re.search(f'({RE_FLOAT})', raw_perf).group(1))

        if len(list(filter(lambda x: x is None, data.values()))) > 0:
            raise ValueError(f'parsing error in {out_filepath} {data}')

        return (cli, [data])

def parse_cholesky(out_filepath):
    with open(out_filepath, 'r') as out_file:
        cli = out_file.readline()[:-1]

        nranks, cores_per_rank = parse_cli(cli)
        
        fields = [
            'run_index',
            'm',
            'mb',
            'grid_rows',
            'grid_cols',
            'time',
            'performance',
            ]
        data = { k:v for k,v in zip(fields, len(fields)*[None]) }

        data['run_index'] = 1

        for data_line in out_file.readlines():
            raw_data_line = data_line[:-1]

            if raw_data_line.startswith(r'#+++++'):
                field, raw_value = raw_data_line[7:].split(':')
                if 'P x Q' in field:
                    data['grid_rows'], data['grid_cols'] = map(int, re.search(r'(\d+)\s+x\s+(\d+)\s*.*', raw_value).groups())
                elif 'M x N x K' in field:
                    data['m'], data['n'] = map(int, re.search(r'(\d+)\s+x\s+(\d+)\s*.*', raw_value).groups())
                elif 'MB x NB' in field:
                    data['mb'], = map(int, re.search(r'(\d+)\s+x\s+\d+', raw_value).groups())
            elif raw_data_line.startswith(r'[****] TIME(s)'):
                raw_data = raw_data_line.split(':')
                raw_time = raw_data[0]
                raw_perf = raw_data[2]
                data['time'] = float(re.search(f'({RE_FLOAT})', raw_time).group(1))
                data['performance'] = float(re.search(f'({RE_FLOAT})', raw_perf).group(1))

        if len(list(filter(lambda x: x is None, data.values()))) > 0:
            raise ValueError(f'parsing error in {out_filepath} {data}')

        return (cli, [data])

if __name__ == "__main__":
    for out_filepath in [
            'raw_data/post/cholesky/25731653J20200918_100712/25731653J20200918_100712.out',
            #'raw_data-cholesky/25683790J20200915_110802/25683790J20200915_110802.out',
            #'raw_data-cholesky/25683793J20200915_113005/25683793J20200915_113005.out'
            ]:
        data = parse_cholesky(out_filepath)
        for a in data:
            print(a)
    exit(1)
    for out_filepath in [
            #"raw_data-pre-upgrade/25522778J20200907_194810/25522778J20200907_194810.out",
            #"raw_data-pre-upgrade/25522042J20200907_171617/25522042J20200907_171617.out",
            #"raw_data-pre-upgrade/25522788J20200907_215439/25522788J20200907_215439.out",
            ]:
            data = parse_trsm(out_filepath)
            for a in data:
                print(a)
