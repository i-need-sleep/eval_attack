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

    data_path = f'{uglobals.DATA_DIR}/{args.data_path}'
    data_df = pd.read_csv(data_path)

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

            idx = df['idx'][adv_idx]
            year = data_df['year'][idx]
            mt_sys = data_df['mt_sys'][idx]

            disp_adv += [
                    dash.html.Div(children=[f'year: {str(year)}']),
                    dash.html.Div(children=[f'system: {mt_sys}']),
                    dash.html.Div(children=f'score: {df["original_score"][adv_idx]} --> {df["adv_score"][adv_idx]}'),
                    dash.html.Div(children=['ref: ', df['ref'][adv_idx]]),
                    dash.html.Div(children=['mt: '] + spans_mt),
                    dash.html.Div(children=['adv: '] + spans_adv),
                    dash.html.P(children=['       ']),
                ]

            # Randomly order the two choices
            lower = 'original'
            if random.random() > 0.5:
                lower = 'adv'
            
            disp_adv_hidden = [
                dash.html.Div(children=['Ref: ']),
                dash.html.Div(children=[df['ref'][adv_idx]]),
                dash.html.Div(children=['Translated:']),
                dash.html.Div(children=[df['adv'][adv_idx]]),
                dash.html.P(children=['       ']),
            ]

            disp_original = [
                dash.html.Div(children=['Ref: ']),
                dash.html.Div(children=[df['ref'][adv_idx]]),
                dash.html.Div(children=['Translated:']),
                dash.html.Div(children=[df['mt'][adv_idx]]),
                dash.html.P(children=['       ']),
            ]
            
            disp_original_full = disp_original

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
    parser.add_argument('--data_path', default='processed/aggregated_de-en_bleurt-20-d12.csv', type=str)

    parser.add_argument('--file_path', default='5-16/20-d12_clare_aggregated_de-en_bleurt-20-d12_bleurt_bleurt-20-d12_down_1.0_gpt10.0_sbert0.9.csv', type=str)

    args = parser.parse_args()

    visualise(args)