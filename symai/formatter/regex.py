import re
import os


# Define variables for magic numbers
MAX_HEADING_LENGTH = 7
MAX_HEADING_CONTENT_LENGTH = 200
MAX_HEADING_UNDERLINE_LENGTH = 200
MAX_HTML_HEADING_ATTRIBUTES_LENGTH = 100
MAX_LIST_ITEM_LENGTH = 200
MAX_NESTED_LIST_ITEMS = 6
MAX_LIST_INDENT_SPACES = 7
MAX_BLOCKQUOTE_LINE_LENGTH = 200
MAX_BLOCKQUOTE_LINES = 15
MAX_CODE_BLOCK_LENGTH = 1500
MAX_CODE_LANGUAGE_LENGTH = 20
MAX_INDENTED_CODE_LINES = 20
MAX_TABLE_CELL_LENGTH = 200
MAX_TABLE_ROWS = 20
MAX_HTML_TABLE_LENGTH = 2000
MIN_HORIZONTAL_RULE_LENGTH = 3
MAX_SENTENCE_LENGTH = 400
MAX_QUOTED_TEXT_LENGTH = 300
MAX_PARENTHETICAL_CONTENT_LENGTH = 200
MAX_NESTED_PARENTHESES = 5
MAX_MATH_INLINE_LENGTH = 100
MAX_MATH_BLOCK_LENGTH = 500
MAX_PARAGRAPH_LENGTH = 1000
MAX_STANDALONE_LINE_LENGTH = 800
MAX_HTML_TAG_ATTRIBUTES_LENGTH = 100
MAX_HTML_TAG_CONTENT_LENGTH = 1000
LOOKAHEAD_RANGE = 100  # Number of characters to look ahead for a sentence boundary

# Define emoji ranges
def generate_emoji_pattern(file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, file_name)

    emoji_codes = set()

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Skip comments and empty lines
                if line.strip() and not line.startswith('#'):
                    fields = line.strip().split(';')
                    unicode_codes = fields[0].strip().split()

                    if len(unicode_codes) == 1:
                        # Single Unicode character
                        emoji_codes.add(chr(int(unicode_codes[0], 16)))
                    elif len(unicode_codes) > 1:
                        # Sequence of Unicode characters
                        emoji_sequence = ''.join(chr(int(code, 16)) for code in unicode_codes)
                        emoji_codes.add(emoji_sequence)

                    # We could also process vendor-specific codes here if needed
                    # for i, vendor_code in enumerate(fields[1:], start=1):
                    #     if vendor_code:
                    #         # Process vendor-specific code
                    #         pass

    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found in {current_dir}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    # Sort the emoji codes
    sorted_codes = sorted(emoji_codes, key=lambda x: [ord(c) for c in x])

    # Generate the regex pattern
    pattern_parts = []
    for code in sorted_codes:
        if len(code) == 1:
            pattern_parts.append(f'\\U{ord(code):08x}')
        else:
            pattern_parts.append(''.join(f'\\U{ord(c):08x}' for c in code))

    emoji_pattern = '(?:' + '|'.join(pattern_parts) + ')'

    return emoji_pattern

# Usage
file_name = 'emoji.pytxt'
EMOJI_PATTERN = generate_emoji_pattern(file_name)

# Define the regex pattern
CHUNK_REGEX = re.compile(
    r"(" +
    # 1. Headings (Setext-style, Markdown, and HTML-style, with length constraints)
    fr"(?:^(?:[#*=-]{{1,{MAX_HEADING_LENGTH}}}|\w[^\r\n]{{0,{MAX_HEADING_CONTENT_LENGTH}}}\r?\n[-=]{{2,{MAX_HEADING_UNDERLINE_LENGTH}}}|<h[1-6][^>]{{0,{MAX_HTML_HEADING_ATTRIBUTES_LENGTH}}}>)[^\r\n]{{1,{MAX_HEADING_CONTENT_LENGTH}}}(?:</h[1-6]>)?(?:\r?\n|$))" +
    "|" +
    # New pattern for citations
    fr"(?:\[[0-9]+\][^\r\n]{{1,{MAX_STANDALONE_LINE_LENGTH}}})" +
    "|" +
    # 2. List items (bulleted, numbered, lettered, or task lists, including nested, up to three levels, with length constraints)
    fr"(?:(?:^|\r?\n)[ \t]{{0,3}}(?:[-*+•]|\d{{1,3}}\.|\[[ xX]\])[ \t]+" +
    fr"(?:(?:\b[^\r\n]{{1,{MAX_LIST_ITEM_LENGTH}}}\b(?:[.!?…]|\.{{3}}|\u2026\u2047-\u2049|{EMOJI_PATTERN})(?=\s|$))|" +
    fr"(?:\b[^\r\n]{{1,{MAX_LIST_ITEM_LENGTH}}}\b(?=[\r\n]|$))|" +
    fr"(?:\b[^\r\n]{{1,{MAX_LIST_ITEM_LENGTH}}}\b(?=[.!?…]|\.{{3}}|\u2026\u2047-\u2049|{EMOJI_PATTERN})" +
    fr"(?:.{{1,{LOOKAHEAD_RANGE}}}(?:[.!?…]|\.{{3}}|\u2026\u2047-\u2049|{EMOJI_PATTERN})(?=\s|$))?))" +
    fr"(?:(?:\r?\n[ \t]{{2,5}}(?:[-*+•]|\d{{1,3}}\.|\[[ xX]\])[ \t]+" +
    fr"(?:(?:\b[^\r\n]{{1,{MAX_LIST_ITEM_LENGTH}}}\b(?:[.!?…]|\.{{3}}|\u2026\u2047-\u2049|{EMOJI_PATTERN})(?=\s|$))|" +
    fr"(?:\b[^\r\n]{{1,{MAX_LIST_ITEM_LENGTH}}}\b(?=[\r\n]|$))|" +
    fr"(?:\b[^\r\n]{{1,{MAX_LIST_ITEM_LENGTH}}}\b(?=[.!?…]|\.{{3}}|\u2026\u2047-\u2049|{EMOJI_PATTERN})" +
    fr"(?:.{{1,{LOOKAHEAD_RANGE}}}(?:[.!?…]|\.{{3}}|\u2026\u2047-\u2049|{EMOJI_PATTERN})(?=\s|$))?))" +
    fr"){{0,{MAX_NESTED_LIST_ITEMS}}}" +
    fr"(?:\r?\n[ \t]{{4,{MAX_LIST_INDENT_SPACES}}}(?:[-*+•]|\d{{1,3}}\.|\[[ xX]\])[ \t]+" +
    fr"(?:(?:\b[^\r\n]{{1,{MAX_LIST_ITEM_LENGTH}}}\b(?:[.!?…]|\.{{3}}|\u2026\u2047-\u2049|{EMOJI_PATTERN})(?=\s|$))|" +
    fr"(?:\b[^\r\n]{{1,{MAX_LIST_ITEM_LENGTH}}}\b(?=[\r\n]|$))|" +
    fr"(?:\b[^\r\n]{{1,{MAX_LIST_ITEM_LENGTH}}}\b(?=[.!?…]|\.{{3}}|\u2026\u2047-\u2049|{EMOJI_PATTERN})" +
    fr"(?:.{{1,{LOOKAHEAD_RANGE}}}(?:[.!?…]|\.{{3}}|\u2026\u2047-\u2049|{EMOJI_PATTERN})(?=\s|$))?))" +
    fr"){{0,{MAX_NESTED_LIST_ITEMS}}})" +
    ")" +
    # 3. Block quotes (including nested quotes and citations, up to three levels, with length constraints)
    fr"(?:(?:^>(?:>|\s{{2,}}){{0,2}}(?:(?:\b[^\r\n]{{0,{MAX_BLOCKQUOTE_LINE_LENGTH}}}\b(?:[.!?…]|\.{{3}}|\u2026\u2047-\u2049|{EMOJI_PATTERN})(?=\s|$))|(?:\b[^\r\n]{{0,{MAX_BLOCKQUOTE_LINE_LENGTH}}}\b(?=[\r\n]|$))|(?:\b[^\r\n]{{0,{MAX_BLOCKQUOTE_LINE_LENGTH}}}\b(?=[.!?…]|\.{{3}}|\u2026\u2047-\u2049|{EMOJI_PATTERN})(?:.{{1,{LOOKAHEAD_RANGE}}}(?:[.!?…]|\.{{3}}|\u2026\u2047-\u2049|{EMOJI_PATTERN})(?=\s|$))?))\\r?\\n?){{1,{MAX_BLOCKQUOTE_LINES}}})" +
    "|" +
    # 4. Code blocks (fenced, indented, or HTML pre/code tags, with length constraints)
    fr"(?:(?:^|\r?\n)(?:```|~~~)(?:\w{{0,{MAX_CODE_LANGUAGE_LENGTH}}})?\r?\n[\s\S]{{0,{MAX_CODE_BLOCK_LENGTH}}}?(?:```|~~~)\r?\n?" +
    fr"|(?:(?:^|\r?\n)(?: {{4}}|\t)[^\r\n]{{0,{MAX_LIST_ITEM_LENGTH}}}(?:\r?\n(?: {{4}}|\t)[^\r\n]{{0,{MAX_LIST_ITEM_LENGTH}}}){{0,{MAX_INDENTED_CODE_LINES}}}\r?\n?)" +
    fr"|(?:<pre>(?:<code>)?[\s\S]{{0,{MAX_CODE_BLOCK_LENGTH}}}?(?:</code>)?</pre>))" +
    "|" +
    # 5. Tables (Markdown, grid tables, and HTML tables, with length constraints)
    fr"(?:(?:^|\r?\n)(?:\|[^\r\n]{{0,{MAX_TABLE_CELL_LENGTH}}}\|(?:\r?\n\|[-:]{{1,{MAX_TABLE_CELL_LENGTH}}}\|){{0,1}}(?:\r?\n\|[^\r\n]{{0,{MAX_TABLE_CELL_LENGTH}}}\|){{0,{MAX_TABLE_ROWS}}}" +
    fr"|<table>[\s\S]{{0,{MAX_HTML_TABLE_LENGTH}}}?</table>))" +
    "|" +
    # 6. Horizontal rules (Markdown and HTML hr tag)
    fr"(?:^(?:[-*_]){{{MIN_HORIZONTAL_RULE_LENGTH},}}\s*$|<hr\s*/?>\r?\n)" +
    "|" +
    # 7. Standalone lines or phrases (including single-line blocks and HTML elements, with length constraints)
    fr"(?:^(?:<[a-zA-Z][^>]{{0,{MAX_HTML_TAG_ATTRIBUTES_LENGTH}}}>)?(?:(?:[^\r\n]{{1,{MAX_STANDALONE_LINE_LENGTH}}}(?:[.!?…]|\.\.\.|\u2026\u2047-\u2049|{EMOJI_PATTERN})(?=\s|$))|(?:[^\r\n]{{1,{MAX_STANDALONE_LINE_LENGTH}}}(?=[\r\n]|$))|(?:[^\r\n]{{1,{MAX_STANDALONE_LINE_LENGTH}}}(?=[.!?…]|\.\.\.|\u2026\u2047-\u2049|{EMOJI_PATTERN})(?:.{{1,{LOOKAHEAD_RANGE}}}(?:[.!?…]|\.\.\.|\u2026\u2047-\u2049|{EMOJI_PATTERN})(?=\s|$))?))(?:</[a-zA-Z]+>)?(?:\r?\n|$))" +
    "|" +
    # 8. Sentences or phrases ending with punctuation (including ellipsis and Unicode punctuation)
    fr"(?:(?:[^\r\n]{{1,{MAX_SENTENCE_LENGTH}}}(?:[.!?…]|\.\.\.|\u2026\u2047-\u2049|{EMOJI_PATTERN})(?=\s|$))|(?:[^\r\n]{{1,{MAX_SENTENCE_LENGTH}}}(?=[\r\n]|$))|(?:[^\r\n]{{1,{MAX_SENTENCE_LENGTH}}}(?=[.!?…]|\.\.\.|\u2026\u2047-\u2049|{EMOJI_PATTERN})(?:.{{1,{LOOKAHEAD_RANGE}}}(?:[.!?…]|\.\.\.|\u2026\u2047-\u2049|{EMOJI_PATTERN})(?=\s|$))?))" +
    "|" +
    # 9. Quoted text, parenthetical phrases, or bracketed content (with length constraints)
    r"(?:" +
    fr'(?<!\w)\"\"\"[^\""]{{0,{MAX_QUOTED_TEXT_LENGTH}}}\"\"\"(?!\w)' +
    fr"|(?<!\w)['\"`''""][^\r\n]{{0,{MAX_QUOTED_TEXT_LENGTH}}}['\"`''""](?!\w)" +
    fr"|\([^\r\n(){{0,{MAX_PARENTHETICAL_CONTENT_LENGTH}}}(?:\([^\r\n(){{0,{MAX_PARENTHETICAL_CONTENT_LENGTH}}}\)[^\r\n(){{0,{MAX_PARENTHETICAL_CONTENT_LENGTH}}}){{0,{MAX_NESTED_PARENTHESES}}}\)" +
    fr"|\[[^\r\n\[\]{{0,{MAX_PARENTHETICAL_CONTENT_LENGTH}}}(?:\[[^\r\n\[\]{{0,{MAX_PARENTHETICAL_CONTENT_LENGTH}}}\][^\r\n\[\]{{0,{MAX_PARENTHETICAL_CONTENT_LENGTH}}}){{0,{MAX_NESTED_PARENTHESES}}}\]" +
    fr"|\$[^\r\n$]{{0,{MAX_MATH_INLINE_LENGTH}}}\$" +
    fr"|`[^`\r\n]{{0,{MAX_MATH_INLINE_LENGTH}}}`" +
    r")" +
    "|" +
    # 10. Paragraphs (with length constraints)
    fr"(?:(?:^|\r?\n\r?\n)(?:<p>)?(?:(?:[^\r\n]{{1,{MAX_PARAGRAPH_LENGTH}}}(?:[.!?…]|\.{{3}}|\u2026\u2047-\u2049|{EMOJI_PATTERN})(?=\s|$))|(?:[^\r\n]{{1,{MAX_PARAGRAPH_LENGTH}}}(?=[\r\n]|$))|(?:[^\r\n]{{1,{MAX_PARAGRAPH_LENGTH}}}(?=[.!?…]|\.{{3}}|\u2026\u2047-\u2049|{EMOJI_PATTERN})(?:.{{1,{LOOKAHEAD_RANGE}}}(?:[.!?…]|\.{{3}}|\u2026\u2047-\u2049|{EMOJI_PATTERN})(?=\s|$))?))(?:</p>)?(?=\r?\n\r?\n|$))" +
    "|" +
    # 11. HTML-like tags and their content (including self-closing tags and attributes, with length constraints)
    fr"(?:<[a-zA-Z][^>]{{0,{MAX_HTML_TAG_ATTRIBUTES_LENGTH}}}(?:>[\s\S]{{0,{MAX_HTML_TAG_CONTENT_LENGTH}}}?</[a-zA-Z]+>|\s*/>))" +
    "|" +
    # 12. LaTeX-style math expressions (inline and block, with length constraints)
    fr"(?:(?:\$\$[\s\S]{{0,{MAX_MATH_BLOCK_LENGTH}}}?\$\$)|(?:\$[^\$\r\n]{{0,{MAX_MATH_INLINE_LENGTH}}}\$))" +
    "|" +
    # 13. Fallback for any remaining content (with length constraints)
    fr"(?:(?:[^\r\n]{{1,{MAX_STANDALONE_LINE_LENGTH}}}(?:[.!?…]|\.{{3}}|\u2026\u2047-\u2049|{EMOJI_PATTERN})(?=\s|$))|(?:[^\r\n]{{1,{MAX_STANDALONE_LINE_LENGTH}}}(?=[\r\n]|$))|(?:[^\r\n]{{1,{MAX_STANDALONE_LINE_LENGTH}}}(?=[.!?…]|\.{{3}}|\u2026\u2047-\u2049|{EMOJI_PATTERN})(?:.{{1,{LOOKAHEAD_RANGE}}}(?:[.!?…]|\.{{3}}|\u2026\u2047-\u2049|{EMOJI_PATTERN})(?=\s|$))?))" +
    ")",
    re.MULTILINE | re.UNICODE
)
