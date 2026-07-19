import re

import pandas as pd
from bs4 import BeautifulSoup, NavigableString, Tag


def clean_images(soup):
    for img in soup.find_all('img'):
        alt = img.get('alt', '').strip()
        if alt:
            img.replace_with(alt)
        else:
            img.decompose()


def clean_breaks(soup):
    for br in soup.find_all('br'):
        br.replace_with(' ')


def clean_inputs(soup):
    for inp in soup.find_all('input'):
        inp.decompose()


def clean_empty_labels(soup):
    for label in soup.find_all('label'):
        text = label.get_text(strip=True)
        if not text:
            label.decompose()


def clean_links(soup):
    for a in soup.find_all('a'):
        text = a.get_text()
        a.replace_with(text)


def clean_headline_chunk(soup):
    for tag in soup.find_all(['headline', 'chunk']):
        tag.unwrap()


def clean_empty_tags(soup):
    for tag in soup.find_all(['p', 'div', 'span', 'strong', 'em']):
        if not tag.get_text(strip=True):
            tag.decompose()


def clean_factoids(soup):
    for factoid in soup.find_all('div', class_=lambda c: c and 'factoid' in c.split()):
        factoid.unwrap()


def clean_spoilers(soup):
    for spoiler in list(soup.find_all('div', class_='spoiler')):
        title_tag = spoiler.select_one('.spoiler-text')
        content_tag = spoiler.select_one('.spoiler-content')

        title = title_tag.get_text(strip=True) if title_tag else ''

        if not content_tag:
            spoiler.decompose()
            continue

        children = list(content_tag.children)
        if not children:
            spoiler.decompose()
            continue

        for child in children:
            child.extract()
            spoiler.insert_before(child)

        if title:
            p = soup.new_tag('p')
            strong = soup.new_tag('strong')
            strong.string = title
            p.append(strong)
            p.append(': ')
            children[0].insert_before(p)

        spoiler.decompose()


def clean_tabsets(soup):
    for tabset in list(
        soup.find_all('div', class_=lambda c: c and 'tabset' in c.split())
    ):
        labels = tabset.find_all('label', class_='tab-label')
        panels = tabset.find_all('div', class_='tab-panel')

        if not panels:
            tabset.unwrap()
            continue

        for i, panel in enumerate(panels):
            title = labels[i].get_text(strip=True) if i < len(labels) else ''
            children = list(panel.children)

            if children:
                for child in children:
                    child.extract()
                    tabset.insert_before(child)

                if title:
                    p = soup.new_tag('p')
                    strong = soup.new_tag('strong')
                    strong.string = title
                    p.append(strong)
                    p.append(': ')
                    children[0].insert_before(p)

        tabset.decompose()


def clean_nested_tables(soup):
    for table in list(soup.find_all('table')):
        while True:
            nested = table.find('table')
            if not nested:
                break
            text = nested.get_text(' ', strip=True)
            nested.replace_with(text)


def clean_empty_tables(soup):
    for table in soup.find_all('table'):
        if not table.get_text(strip=True):
            table.decompose()


PIPELINE = [
    ('clean_images', clean_images),
    ('clean_breaks', clean_breaks),
    ('clean_inputs', clean_inputs),
    ('clean_empty_labels', clean_empty_labels),
    ('clean_links', clean_links),
    ('clean_headline_chunk', clean_headline_chunk),
    ('clean_empty_tags', clean_empty_tags),
    ('clean_factoids', clean_factoids),
    ('clean_spoilers', clean_spoilers),
    ('clean_tabsets', clean_tabsets),
    ('clean_nested_tables', clean_nested_tables),
    ('clean_empty_tables', clean_empty_tables),
]


def preprocess_html(html_text: str) -> str:
    soup = BeautifulSoup(html_text, 'html.parser')

    for name, func in PIPELINE:
        func(soup)

    body = soup.body
    if body:
        return ''.join(str(child) for child in body.children)
    return str(soup)


def preprocess_dataframe(
    df: pd.DataFrame, column: str = 'body'
) -> pd.DataFrame:
    result = df.copy()
    result[column] = result[column].apply(preprocess_html)
    return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Preprocess HTML articles: handle Avito-specific markup'
    )
    parser.add_argument('input', help='Path to input .feather file')
    parser.add_argument(
        '--column', default='body', help='Column with HTML text (default: body)'
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
    before = len(df)

    df = preprocess_dataframe(df, column=args.column)

    df.to_feather(output_path)
    print(f'Processed {len(df)} rows, saved to {output_path}')
