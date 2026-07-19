import re

import html2text
import pandas as pd


HTML2TEXT_CONFIG = {
    'body_width': 0,
    'ignore_emphasis': True,
    'ignore_links': True,
    'ignore_images': True,
    'single_line_break': True,
    'pad_tables': True,
    'unicode_snob': True,
}


def build_converter() -> html2text.HTML2Text:
    h = html2text.HTML2Text()
    for key, value in HTML2TEXT_CONFIG.items():
        setattr(h, key, value)
    return h


def collapse_whitespace(text: str) -> str:
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    return text


def convert_html_to_text(html: str) -> str:
    h = build_converter()
    text = h.handle(html)
    text = collapse_whitespace(text)
    return text


def convert_dataframe(
    df: pd.DataFrame, column: str = 'body', output_column: str = 'body'
) -> pd.DataFrame:
    result = df.copy()
    result[output_column] = result[column].apply(convert_html_to_text)
    return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Stage 2: Convert cleaned HTML to Markdown/text via html2text'
    )
    parser.add_argument('input', help='Path to input .feather file')
    parser.add_argument(
        '--column', default='body', help='Input column with HTML (default: body)'
    )
    parser.add_argument(
        '--output-column',
        default='body',
        help='Output column name (default: body)',
    )
    parser.add_argument(
        'output',
        nargs='?',
        default=None,
        help='Path to output .feather file (default: overwrite input)',
    )

    args = parser.parse_args()
    output_path = args.output or args.input

    df = pd.read_feather(args.input)
    df = convert_dataframe(df, column=args.column, output_column=args.output_column)
    df.to_feather(output_path)
    print(
        f'Converted {len(df)} articles, saved to {output_path}'
    )
