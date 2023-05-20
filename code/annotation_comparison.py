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
    'text_highlight': 'red',
    'text': 'black'
}

def visualise(args):
    app = dash.Dash(__name__)

    file_path = f'{OUTPUT_DIR}/{args.file_path}'
    df = pd.read_csv(file_path)

    data_path = f'{uglobals.DATA_DIR}/{args.data_path}'
    data_df = pd.read_csv(data_path)

    # Shuffle the df
    df = df.sample(frac=1)

    # Setup the page layout
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
            def next_pair():
                global disp_0_full, disp_1_full

                # Choose the comparison type: mt-mt, mt-adv, or adv-adv
                comparison_type = random.choice(['mt-mt', 'mt-adv', 'adv-adv'])

                if comparison_type == 'mt-mt':
                    # Make a shuffled copy of the metadata, keeping the original copy unchanged since we need it for the metadata for adv cases.
                    shuffled_data_df = data_df.sample(frac=1)

                    # Draw an mt instance
                    mt1_idx = random.choice(range(len(shuffled_data_df)))
                    
                    # Find an mt instance with an acceptable score difference
                    mt2_idx = -1
                    for i in range(len(shuffled_data_df)):
                        if i == mt1_idx:
                            continue
                        if abs(shuffled_data_df['normalized_score'][mt1_idx] - shuffled_data_df['normalized_score'][i]) < args.max_score_difference:
                            mt2_idx = i
                            break

                    # In the case where no acceptable other mt instance exist
                    if mt2_idx == -1:
                        return next_pair()
                    
                    # Make displayed texts
                    disp_mt1 = [
                        dash.html.Div(children=['Ref: ']),
                        dash.html.Div(children=[shuffled_data_df['ref'][mt1_idx]]),
                        dash.html.Div(children=['Translated: ']),
                        dash.html.Div(children=[shuffled_data_df['mt'][mt1_idx]]),
                        dash.html.P(children=['       ']),
                    ]
                    disp_mt2 = [
                        dash.html.Div(children=['Ref: ']),
                        dash.html.Div(children=[shuffled_data_df['ref'][mt2_idx]]),
                        dash.html.Div(children=['Translated: ']),
                        dash.html.Div(children=[shuffled_data_df['mt'][mt2_idx]]),
                        dash.html.P(children=['       ']),
                    ]

                    # Hidden texts with metadata
                    if shuffled_data_df["normalized_score"][mt1_idx] > shuffled_data_df["normalized_score"][mt2_idx]:
                        color_1 = COLOR['text_highlight']
                        color_2 = COLOR['text']
                    else:
                        color_1 = COLOR['text']
                        color_2 = COLOR['text_highlight']
                    
                        
                    disp_mt1_full = [
                        dash.html.Div(children=[f'year: {shuffled_data_df["year"][mt1_idx]}']),
                        dash.html.Div(children=[f'system: {shuffled_data_df["mt_sys"][mt1_idx]}']),
                        dash.html.Div(children=f'score: {shuffled_data_df["normalized_score"][mt1_idx]}', style={'color': color_1}),
                        dash.html.Div(children=['ref: ', shuffled_data_df['ref'][mt1_idx]]),
                        dash.html.Div(children=['translated:', shuffled_data_df['mt'][mt1_idx]]),
                        dash.html.P(children=['       ']),
                    ]
                    disp_mt2_full = [
                        dash.html.Div(children=[f'year: {shuffled_data_df["year"][mt2_idx]}']),
                        dash.html.Div(children=[f'system: {shuffled_data_df["mt_sys"][mt2_idx]}']),
                        dash.html.Div(children=f'score: {shuffled_data_df["normalized_score"][mt2_idx]}', style={'color': color_2}),
                        dash.html.Div(children=['ref: ', shuffled_data_df['ref'][mt2_idx]]),
                        dash.html.Div(children=['translated:', shuffled_data_df['mt'][mt2_idx]]),
                        dash.html.P(children=['       ']),
                    ]
                    
                    if random.random() > 0.5:
                        disp_0 = disp_mt1
                        disp_1 = disp_mt2
                        disp_0_full = disp_mt1_full
                        disp_1_full = disp_mt2_full
                    else:
                        disp_1 = disp_mt2
                        disp_0 = disp_mt1
                        disp_1_full = disp_mt2_full
                        disp_0_full = disp_mt1_full

                    return disp_0 + disp_1
                    
                elif comparison_type == 'mt-adv':
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
                        dash.html.Div(children=f'score: {df["original_score"][adv_idx]} --> {df["adv_score"][adv_idx]}', style={'color': COLOR['text_highlight']}),
                        dash.html.Div(children=['ref: ', df['ref'][adv_idx]]),
                        dash.html.Div(children=['mt: '] + spans_mt),
                        dash.html.Div(children=['adv: '] + spans_adv),
                        dash.html.P(children=['       ']),
                    ]


                    # Find the translation with a score within the specified range
                    original_idx = -1
                    lower = 'original'
                    shuffled_data_df = data_df.sample(frac=1)
                    if random.random() > 0.5:
                        # find an original translation with a higher score
                        lower = 'adv'
                        for i in range(len(shuffled_data_df)):
                            original_line = shuffled_data_df.iloc[i]
                            score_diff = original_line['normalized_score'] - df["adv_score"][adv_idx]
                            if i != idx and score_diff > 0 and abs(score_diff) < args.max_score_difference:
                                original_idx = i
                                line_out = original_line
                                break
                    else:
                        # find an original translation with a lower score
                        for i in range(len(shuffled_data_df)):
                            original_line = shuffled_data_df.iloc[i]
                            score_diff = df["adv_score"][adv_idx] - original_line['normalized_score']
                            if i != idx and score_diff > 0 and abs(score_diff) < args.max_score_difference:
                                original_idx = i
                                line_out = original_line
                                break

                    # Exception case
                    if original_idx == -1:
                        return next_pair()

                    original_line = line_out
                    
                    disp_adv_hidden = [
                        dash.html.Div(children=['Ref: ']),
                        dash.html.Div(children=[df['ref'][adv_idx]]),
                        dash.html.Div(children=['Translated:']),
                        dash.html.Div(children=[df['adv'][adv_idx]]),
                        dash.html.P(children=['       ']),
                    ]

                    disp_original = [
                        dash.html.Div(children=['Ref: ']),
                        dash.html.Div(children=[original_line['ref']]),
                        dash.html.Div(children=['Translated: ']),
                        dash.html.Div(children=[original_line['mt']]),
                        dash.html.P(children=['       ']),
                    ]
                    
                    if lower == 'original':
                        disp_original_full = [
                            dash.html.Div(children=[f'year: {original_line["year"]}']),
                            dash.html.Div(children=[f'system: {original_line["mt_sys"]}']),
                            dash.html.Div(children=f'score: {original_line["normalized_score"]}'),
                            dash.html.Div(children=['ref: ', original_line['ref']]),
                            dash.html.Div(children=['translated:', original_line['mt']]),
                            dash.html.P(children=['       ']),
                        ]
                    else:
                        disp_original_full = [
                            dash.html.Div(children=[f'year: {original_line["year"]}']),
                            dash.html.Div(children=[f'system: {original_line["mt_sys"]}']),
                            dash.html.Div(children=f'score: {original_line["normalized_score"]}', style={'color': COLOR['text_highlight']}),
                            dash.html.Div(children=['ref: ', original_line['ref']]),
                            dash.html.Div(children=['translated:', original_line['mt']]),
                            dash.html.P(children=['       ']),
                        ]

                        disp_adv = [
                            dash.html.Div(children=[f'year: {str(year)}']),
                            dash.html.Div(children=[f'system: {mt_sys}']),
                            dash.html.Div(children=f'score: {df["original_score"][adv_idx]} --> {df["adv_score"][adv_idx]}'),
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
                
                elif comparison_type == 'adv-adv':
                    # Draw an adv and make highlights
                    adv_idx = random.choice(range(len(df)))
                    disp_adv = []
                    spans_mt, spans_adv = make_highlight_spans(df['mt'][adv_idx].split(' '), df['adv'][adv_idx].split(' '))

                    idx = df['idx'][adv_idx]
                    year = data_df['year'][idx]
                    mt_sys = data_df['mt_sys'][idx]

                    # Find a nother adv instance where the score difference falls with in the specified range
                    adv2_idx = -1
                    for i in range(len(df)):
                        if abs(df['adv_score'][i] - df['adv_score'][adv_idx]) < args.max_score_difference:
                            adv2_idx = i
                            break

                    # Exception case
                    if adv2_idx == -1:
                        return next_pair()
                    
                    if df['adv_score'][i] - df['adv_score'][adv_idx] < 0:
                        color_1 = COLOR['text_highlight']
                        color_2 = COLOR['text']
                    else:
                        color_1 = COLOR['text']
                        color_2 = COLOR['text_highlight']
                    
                    disp_adv = [
                        dash.html.Div(children=['Ref: ']),
                        dash.html.Div(children=[df['ref'][adv_idx]]),
                        dash.html.Div(children=['Translated:']),
                        dash.html.Div(children=[df['adv'][adv_idx]]),
                        dash.html.P(children=['       ']),
                    ]

                    disp_adv2 = [
                        dash.html.Div(children=['Ref: ']),
                        dash.html.Div(children=[df['ref'][adv2_idx]]),
                        dash.html.Div(children=['Translated:']),
                        dash.html.Div(children=[df['adv'][adv2_idx]]),
                        dash.html.P(children=['       ']),
                    ]

                    disp_adv_full = [
                        dash.html.Div(children=[f'year: {str(year)}']),
                        dash.html.Div(children=[f'system: {mt_sys}']),
                        dash.html.Div(children=f'score: {df["original_score"][adv_idx]} --> {df["adv_score"][adv_idx]}', style={'color': color_1}),
                        dash.html.Div(children=['ref: ', df['ref'][adv_idx]]),
                        dash.html.Div(children=['mt: '] + spans_mt),
                        dash.html.Div(children=['adv: '] + spans_adv),
                        dash.html.P(children=['       ']),
                    ]

                    spans_mt, spans_adv = make_highlight_spans(df['mt'][adv2_idx].split(' '), df['adv'][adv2_idx].split(' '))
                    idx = df['idx'][adv2_idx]
                    year = data_df['year'][idx]
                    mt_sys = data_df['mt_sys'][idx]

                    disp_adv2_full = [
                        dash.html.Div(children=[f'year: {str(year)}']),
                        dash.html.Div(children=[f'system: {mt_sys}']),
                        dash.html.Div(children=f'score: {df["original_score"][adv2_idx]} --> {df["adv_score"][adv2_idx]}', style={'color': color_2}),
                        dash.html.Div(children=['ref: ', df['ref'][adv2_idx]]),
                        dash.html.Div(children=['mt: '] + spans_mt),
                        dash.html.Div(children=['adv: '] + spans_adv),
                        dash.html.P(children=['       ']),
                    ]

                    if random.random() > 0.5:
                        disp_0 = disp_adv
                        disp_1 = disp_adv2
                        disp_0_full = disp_adv_full
                        disp_1_full = disp_adv2_full
                    else:
                        disp_0 = disp_adv2
                        disp_1 = disp_adv
                        disp_0_full = disp_adv2_full
                        disp_1_full = disp_adv_full

                    return disp_0 + disp_1
                
                else:
                    raise NotImplementedError
            
            return next_pair()

        if ctx.triggered_id == 'button_show':
            return disp_0_full + disp_1_full
    

    app.run_server(debug=False)
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

    # Comparison rules
    parser.add_argument('--max_score_difference', default=0.1, type=float)

    args = parser.parse_args()

    visualise(args)