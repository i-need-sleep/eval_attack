import argparse

import dash
import pandas as pd
import Levenshtein

OUTPUT_DIR = '../results/outputs'
COLOR = {
    'text_highlight': 'red'
}

def visualise(args):
    app = dash.Dash(__name__)

    file_path = f'{OUTPUT_DIR}/{args.file_path}'

    df = pd.read_csv(file_path)

    content = [
        dash.html.Div(children=file_path),
    ]
    
    for i in range(len(df)):
        if Levenshtein.distance(df['mt'][i].split(' '), df['adv'][i].split(' ')) <= args.min_edit_dist:
            continue
        spans_mt, spans_adv = make_highlight_spans(df['mt'][i].split(' '), df['adv'][i].split(' '))
        content += [
            dash.html.Div(children=f'Score: {df["original_score"][i]} --> {df["adv_score"][i]}'),
            dash.html.Div(children=['mt: '] + spans_mt),
            dash.html.Div(children=['adv: '] + spans_adv),
            dash.html.Div(children=['ref: ', df['ref'][i]]),
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

    parser.add_argument('--file_path', default='poc_2017-da_bleurt_down_0.2_0.2_100_gpt2.csv', type=str) 
    # parser.add_argument('--file_path', default='2-20/poc_wmt-zhen-tedtalks_bleurt_down_0.35_300.csv', type=str) 

    parser.add_argument('--min_edit_dist', default=0, type=int) 

    args = parser.parse_args()

    visualise(args)