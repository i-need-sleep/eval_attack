import argparse
import random

import dash
from dash import Dash, dcc, html, ctx
from dash.dependencies import Input, Output, State
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
        dash.html.P(children=['       ']),
        dash.html.Div(children='', id='texts'),
        dash.html.Button(id='button_show', children='Show', n_clicks=0),
        dash.html.Button(id='button_next', children='Next', n_clicks=0),
    ]
    
    app.layout = dash.html.Div(children=content)
    
    disp_0_full = []
    disp_1_full = []

    @app.callback(Output('texts', 'children'),
              Input('button_show', 'n_clicks'),
              Input('button_next', 'n_clicks'))
    def update_output(_, a):
        if ctx.triggered_id == 'button_next':
            global disp_0_full, disp_1_full

            # Draw an adv and make highlights
            adv_idx = random.choice(range(len(df)))
            disp_adv = []
            spans_mt, spans_adv = make_highlight_spans(df['mt'][adv_idx].split(' '), df['adv'][adv_idx].split(' '))

            if args.data_path != '':
                idx = df['idx'][adv_idx]
                year = data_df['year'][idx]
                mt_sys = data_df['mt_sys'][idx]

                disp_adv += [
                    dash.html.Div(children=[f'year: {str(year)}']),
                    dash.html.Div(children=[f'system: {mt_sys}']),
                ]

            disp_adv += [
                dash.html.Div(children=f'score: {df["original_score"][adv_idx]} --> {df["adv_score"][adv_idx]}'),
                dash.html.Div(children=['ref: ', df['ref'][adv_idx]]),
                dash.html.Div(children=['mt: '] + spans_mt),
                dash.html.Div(children=['adv: '] + spans_adv),
                dash.html.P(children=['       ']),
            ]

            # Find the translation with the closetest score
            original_idx = -1
            best_score = 999
            lower = 'original'
            if random.random() > 0.5:
                # find an original translation with a higher score
                lower = 'adv'
                for i in range(len(data_df)):
                    score_diff = data_df['normalized_score'][i] - df["adv_score"][adv_idx]
                    if i != idx and score_diff > 0:
                        if score_diff < best_score:
                            best_score = score_diff
                            original_idx = i
            else:
                # find an original translation with a lower score
                for i in range(len(data_df)):
                    score_diff = df["adv_score"][adv_idx] - data_df['normalized_score'][i] 
                    if i != idx and score_diff > 0:
                        if score_diff < best_score:
                            best_score = score_diff
                            original_idx = i
            
            disp_adv_hidden = [
                dash.html.Div(children=['Ref: ']),
                dash.html.Div(children=[df['ref'][adv_idx]]),
                dash.html.Div(children=['Translated:']),
                dash.html.Div(children=[df['adv'][adv_idx]]),
                dash.html.P(children=['       ']),
            ]

            disp_original = [
                dash.html.Div(children=['Ref: ']),
                dash.html.Div(children=[data_df['ref'][original_idx]]),
                dash.html.Div(children=['Translated: ']),
                dash.html.Div(children=[data_df['mt'][original_idx]]),
                dash.html.P(children=['       ']),
            ]
            
            if lower == 'original':
                disp_original_full = [
                    dash.html.Div(children=[f'year: {data_df["year"][original_idx]}']),
                    dash.html.Div(children=[f'system: {data_df["mt_sys"][original_idx]}']),
                    dash.html.Div(children=f'score: {data_df["normalized_score"][original_idx]}', style={'color': COLOR['text_highlight']}),
                    dash.html.Div(children=['ref: ', data_df['ref'][original_idx]]),
                    dash.html.Div(children=['translated:', data_df['mt'][original_idx]]),
                    dash.html.P(children=['       ']),
                ]
            else:
                disp_original_full = [
                    dash.html.Div(children=[f'year: {data_df["year"][original_idx]}']),
                    dash.html.Div(children=[f'system: {data_df["mt_sys"][original_idx]}']),
                    dash.html.Div(children=f'score: {data_df["normalized_score"][original_idx]}'),
                    dash.html.Div(children=['ref: ', data_df['ref'][original_idx]]),
                    dash.html.Div(children=['translated:', data_df['mt'][original_idx]]),
                    dash.html.P(children=['       ']),
                ]

                disp_adv = [
                    dash.html.Div(children=[f'year: {str(year)}']),
                    dash.html.Div(children=[f'system: {mt_sys}']),
                    dash.html.Div(children=f'score: {df["original_score"][adv_idx]} --> {df["adv_score"][adv_idx]}', style={'color': COLOR['text_highlight']}),
                    dash.html.Div(children=['ref: ', df['ref'][adv_idx]]),
                    dash.html.Div(children=['mt: '] + spans_mt),
                    dash.html.Div(children=['adv: '] + spans_adv),
                    dash.html.P(children=['       ']),
                ]

            if random.random() > 0.5:
                disp_0 = disp_adv_hidden
                disp_1 = disp_original
                disp_0_full = disp_adv
                disp_1_full = disp_original_full
            else:
                disp_1 = disp_adv_hidden
                disp_0 = disp_original
                disp_1_full = disp_adv
                disp_0_full = disp_original_full

            return disp_0 + disp_1

        if ctx.triggered_id == 'button_show':
            return disp_0_full + disp_1_full
    

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
    parser.add_argument('--file_path', default='3-20/20-d12_clare_aggregated_de-en_bleurt-20-d12_bleurt_bleurt-20-d12_down_0.5_gpt10.0_sbert0.9.csv', type=str)

    # parser.add_argument('--file_path', default='3-20/20-d12_faster_genetic_aggregated_de-en_bleurt-20-d12_bleurt_bleurt-20-d12_down_0.2_gpt10.0_sbert0.9.csv', type=str)
    # parser.add_argument('--file_path', default='3-20/20-d12_faster_genetic_aggregated_de-en_bleurt-20-d12_bleurt_bleurt-20-d12_down_0.5_gpt10.0_sbert0.9.csv', type=str)

    # parser.add_argument('--file_path', default='3-20/20-d12_input_reduction_aggregated_de-en_bleurt-20-d12_bleurt_bleurt-20-d12_down_0.2.csv', type=str)
    # parser.add_argument('--file_path', default='3-20/20-d12_input_reduction_aggregated_de-en_bleurt-20-d12_bleurt_bleurt-20-d12_down_0.5.csv', type=str)

    # parser.add_argument('--data_path', default='processed/aggregated_de-en_bertscore.csv', type=str)

    # parser.add_argument('--file_path', default='3-20/bertscore_clare_aggregated_de-en_bertscore_bertscore__down_0.2_gpt10.0_sbert0.9.csv', type=str)
    # parser.add_argument('--file_path', default='3-20/bertscore_clare_aggregated_de-en_bertscore_bertscore__down_0.5_gpt10.0_sbert0.9.csv', type=str)

    # parser.add_argument('--file_path', default='3-20/bertscore_faster_genetic_aggregated_de-en_bertscore_bertscore__down_0.2_gpt10.0_sbert0.9.csv', type=str)
    # parser.add_argument('--file_path', default='3-20/bertscore_faster_genetic_aggregated_de-en_bertscore_bertscore__down_0.5_gpt10.0_sbert0.9.csv', type=str)

    # parser.add_argument('--file_path', default='3-20/bertscore_input_reduction_aggregated_de-en_bertscore_bertscore__down_0.2.csv', type=str)
    # parser.add_argument('--file_path', default='3-20/bertscore_input_reduction_aggregated_de-en_bertscore_bertscore__down_0.5.csv', type=str)
    
    # Sorting
    parser.add_argument('--min_edit_dist', default=0, type=int) 
    parser.add_argument('--sort_by_diff', action='store_true')

    # Display limit
    parser.add_argument('--max_n_displayed', default=400, type=int) 

    args = parser.parse_args()

    visualise(args)