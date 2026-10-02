import re
import pandas as pd
from embedding.config import MAX_LYRIC_WORDS, MAX_SNIPPET_LENGTH


def clean_lyrics(text: str) -> str:
    
    if not isinstance(text, str):
        return ""

    # remove annotations in brackets or parentheses like [Chorus], (Guitar Solo)
    cleaned = re.sub(r"\[.*?\]", " ", text)
    cleaned = re.sub(r"\(.*?\)", " ", cleaned)

    # replace newlines and tabs with spaces
    cleaned = re.sub(r"[\r\n\t]+", " ", cleaned)

    # multiple spaces to a single space
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def truncate_to_word_budget(text: str, max_words: int = MAX_LYRIC_WORDS) -> str:

    # just keep the input string under or qual to the MAX_LYRIC_WORDS
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def create_search_payload(artist: str, song: str, lyrics: str, max_words: int = MAX_LYRIC_WORDS) -> str:

    # building the string needed to be passed into the embedding model:
    # 'Title: <song> | Artist: <artist> | Lyrics: <first 200 words>'
    cleaned = clean_lyrics(lyrics)
    truncated = truncate_to_word_budget(cleaned, max_words=max_words)
    return f"Title: {song.strip()} | Artist: {artist.strip()} | Lyrics: {truncated}"


def create_lyrics_snippet(lyrics: str, max_chars: int = 240) -> str:

    # just clean the lyrics, (if over the MAX_LYRIC_WORDS, add ...), for displaying on the UI (lyrics-preview) 
    cleaned = clean_lyrics(lyrics)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "..."


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:

    # copy the dataframe
    df = df.copy()

    # Fill missing values defensively so Python never crashes on NaNs
    df["artist"] = df["artist"].fillna("Unknown Artist").astype(str).str.strip()
    df["song"] = df["song"].fillna("Unknown Title").astype(str).str.strip()
    df["link"] = df["link"].fillna("").astype(str).str.strip()
    df["text"] = df["text"].fillna("").astype(str)

    # 1. Create the composite search text for the embedding model
    df["search_text"] = df.apply(
        lambda row: create_search_payload(row["artist"], row["song"], row["text"]),
        axis=1
    )

    # 2. Create the preview snippet for the UI card
    df["lyrics_snippet"] = df["text"].apply(
        lambda val: create_lyrics_snippet(val, max_chars=240)
    )

    return df
