import argparse

import dash
import pandas as pd
import Levenshtein

import utils.globals as uglobals

OUTPUT_DIR = '../results/outputs'
COLOR = {
    'text_highlight': 'red'
}

def visualise(args):
    app = dash.Dash(__name__)

    file_path = f'{OUTPUT_DIR}/{args.file_path}'
    df = pd.read_csv(file_path)

    if args.data_path != '':
        data_path = f'{uglobals.DATA_DIR}/{args.data_path}'
        data_df = pd.read_csv(data_path)

    # Sorting
    if args.sort_by_diff:
        df['diff'] = df["original_score"] - df["adv_score"]
        df = df.sort_values('diff', ascending=False, ignore_index=True)

    content = [
        dash.html.Div(children=file_path),
    ]
    
    for i in range(len(df)):
        if i > args.max_n_displayed:
            break
        if Levenshtein.distance(df['mt'][i].split(' '), df['adv'][i].split(' ')) <= args.min_edit_dist:
            continue
        spans_mt, spans_adv = make_highlight_spans(df['mt'][i].split(' '), df['adv'][i].split(' '))

        # Retrieve metadata
        if args.data_path != '':
            idx = df['idx'][i]
            year = data_df['year'][idx]
            mt_sys = data_df['mt_sys'][idx]

            content += [
                dash.html.Div(children=[f'year: {str(year)}']),
                dash.html.Div(children=[f'system: {mt_sys}']),
            ]


        content += [
            dash.html.Div(children=f'score: {df["original_score"][i]} --> {df["adv_score"][i]}'),
            dash.html.Div(children=['ref: ', df['ref'][i]]),
            dash.html.Div(children=['mt: '] + spans_mt),
            dash.html.Div(children=['adv: '] + spans_adv),
            dash.html.P(children=['       ']),
        ]
    
    app.layout = dash.html.Div(children=content)

    app.run_server(debug=True)
    return

def make_highlight_spans(a, b):
    blocks = Levenshtein.matching_blocks(Levenshtein.editops(a, b), a, b)

    out_a = []
    cur_chars = ''
    a += ' '
    for idx, char in enumerate(a):
        added = False
        for block in blocks:
            # Is it the start of an overlapping span?
            if block[0] == idx:
                # Add a highlighted span
                out_a.append(dash.html.Span(children=cur_chars, style={'color': COLOR['text_highlight']}))
                cur_chars = char + ' '
                added = True
                break
            # End of overlapping span
            elif block[0] + block[2] == idx:
                # Add a regular span
                out_a.append(dash.html.Span(children=cur_chars))
                cur_chars = char + ' '
                added = True
                break
        if added:
            continue
        cur_chars += char + ' '
        if idx == len(a) and cur_chars != char:
            # Add a highlighted span
            out_a.append(dash.html.Span(children=cur_chars, style={'color': COLOR['text_highlight']}))
    
    out_b = []
    cur_chars = ''
    b += ' '
    for idx, char in enumerate(b):
        added = False
        for block in blocks:
            # Is it the start of an overlapping span?
            if block[1] == idx:
                # Add a highlighted span
                out_b.append(dash.html.Span(children=cur_chars, style={'color': COLOR['text_highlight']}))
                cur_chars = char + ' '
                added = True
                break
            # End of overlapping span
            elif block[1] + block[2] == idx:
                # Add a regular span
                out_b.append(dash.html.Span(children=cur_chars))
                cur_chars = char + ' '
                added = True
                break
        if added:
            continue
        cur_chars += char + ' '
        if idx == len(b) and cur_chars != char:
            # Add a highlighted span
            out_b.append(dash.html.Span(children=cur_chars, style={'color': COLOR['text_highlight']}))
            
    return out_a, out_b
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Paths
    # parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--data_path', default='processed/aggregated_de-en_bleurt-20-d12.csv', type=str)

    # parser.add_argument('--file_path', default='3-20/20-d12_clare_aggregated_de-en_bleurt-20-d12_bleurt_bleurt-20-d12_down_0.2_gpt10.0_sbert0.9.csv', type=str)
    # parser.add_argument('--file_path', default='3-20/20-d12_clare_aggregated_de-en_bleurt-20-d12_bleurt_bleurt-20-d12_down_0.5_gpt10.0_sbert0.9.csv', type=str)
    parser.add_argument('--file_path', default='3-20/20-d12_clare_aggregated_de-en_bleurt-20-d12_bleurt_bleurt-20-d12_down_1.0_gpt10.0_sbert0.9.csv', type=str)

    # parser.add_argument('--file_path', default='3-20/20-d12_faster_genetic_aggregated_de-en_bleurt-20-d12_bleurt_bleurt-20-d12_down_0.2_gpt10.0_sbert0.9.csv', type=str)
    # parser.add_argument('--file_path', default='3-20/20-d12_faster_genetic_aggregated_de-en_bleurt-20-d12_bleurt_bleurt-20-d12_down_0.5_gpt10.0_sbert0.9.csv', type=str)
    # parser.add_argument('--file_path', default='3-20/20-d12_faster_genetic_aggregated_de-en_bleurt-20-d12_bleurt_bleurt-20-d12_down_1.0_gpt10.0_sbert0.9.csv', type=str)

    # parser.add_argument('--file_path', default='3-20/20-d12_input_reduction_aggregated_de-en_bleurt-20-d12_bleurt_bleurt-20-d12_down_0.2.csv', type=str)
    # parser.add_argument('--file_path', default='3-20/20-d12_input_reduction_aggregated_de-en_bleurt-20-d12_bleurt_bleurt-20-d12_down_0.5.csv', type=str)
    # parser.add_argument('--file_path', default='3-20/20-d12_input_reduction_aggregated_de-en_bleurt-20-d12_bleurt_bleurt-20-d12_down_1.0.csv', type=str)

    # parser.add_argument('--data_path', default='processed/aggregated_de-en_bertscore.csv', type=str)

    # parser.add_argument('--file_path', default='3-20/bertscore_clare_aggregated_de-en_bertscore_bertscore__down_0.2_gpt10.0_sbert0.9.csv', type=str)
    # parser.add_argument('--file_path', default='3-20/bertscore_clare_aggregated_de-en_bertscore_bertscore__down_0.5_gpt10.0_sbert0.9.csv', type=str)
    # parser.add_argument('--file_path', default='3-20/bertscore_clare_aggregated_de-en_bertscore_bertscore__down_1.0_gpt10.0_sbert0.9.csv', type=str)

    # parser.add_argument('--file_path', default='3-20/bertscore_faster_genetic_aggregated_de-en_bertscore_bertscore__down_0.2_gpt10.0_sbert0.9.csv', type=str)
    # parser.add_argument('--file_path', default='3-20/bertscore_faster_genetic_aggregated_de-en_bertscore_bertscore__down_0.5_gpt10.0_sbert0.9.csv', type=str)
    # parser.add_argument('--file_path', default='3-20/bertscore_faster_genetic_aggregated_de-en_bertscore_bertscore__down_1.0_gpt10.0_sbert0.9.csv', type=str)

    # parser.add_argument('--file_path', default='3-20/bertscore_input_reduction_aggregated_de-en_bertscore_bertscore__down_0.2.csv', type=str)
    # parser.add_argument('--file_path', default='3-20/bertscore_input_reduction_aggregated_de-en_bertscore_bertscore__down_0.5.csv', type=str)
    # parser.add_argument('--file_path', default='3-20/bertscore_input_reduction_aggregated_de-en_bertscore_bertscore__down_1.0.csv', type=str)
    
    # Sorting
    parser.add_argument('--min_edit_dist', default=0, type=int) 
    parser.add_argument('--sort_by_diff', action='store_true')

    # Display limit
    parser.add_argument('--max_n_displayed', default=400, type=int) 

    args = parser.parse_args()

    visualise(args)