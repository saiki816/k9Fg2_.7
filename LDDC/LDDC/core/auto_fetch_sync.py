# SPDX-FileCopyrightText: Copyright (C) 2024-2025 沉默の金 <cmzj@cmzj.org>
# SPDX-License-Identifier: GPL-3.0-only
"""
This module is a refactored version of auto_fetch.py, designed to work in a
non-Qt server environment. It removes all dependencies on PySide6 and TaskManager,
using Python's standard concurrent.futures for asynchronous operations.
"""

from collections.abc import Iterable
from functools import reduce
from typing import Literal, overload
from concurrent.futures import ThreadPoolExecutor, as_completed, Future, wait, ALL_COMPLETED

from LDDC.common.exceptions import AutoFetchUnknownError, LDDCError, LyricsNotFoundError, NotEnoughInfoError
from LDDC.common.logger import logger
from LDDC.common.models import APIResultList, Language, LyricInfo, Lyrics, LyricsType, SearchInfo, SearchType, SongInfo, Source
from LDDC.core.algorithm import calculate_artist_score, calculate_title_score, text_difference
from LDDC.core.api.lyrics import get_lyrics, search


@overload
def auto_fetch(
    info: SongInfo,
    min_score: float = 60,
    sources: Iterable[Source] = (Source.QM, Source.KG, Source.NE),
    return_search_results: bool = False,
) -> Lyrics: ...


@overload
def auto_fetch(
    info: SongInfo,
    min_score: float = 60,
    sources: Iterable[Source] = (Source.QM, Source.KG, Source.NE),
    return_search_results: bool = True,
) -> tuple[Lyrics, APIResultList[SongInfo]]: ...


def auto_fetch(
    info: SongInfo,
    min_score: float = 55,
    sources: Iterable[Source] = (Source.QM, Source.KG, Source.NE),
    return_search_results: bool = False,
    timeout: int = 30,
) -> Lyrics | tuple[Lyrics, APIResultList[SongInfo]]:
    keywords: dict[Literal["artist-title", "title", "file_name"], str] = {}
    if info.title and info.title.strip():
        if info.artist:
            keywords["artist-title"] = info.artist_title()
        keywords["title"] = info.title
    elif info.path:
        keywords["file_name"] = info.path.stem
    else:
        msg = f"没有足够的信息用于搜索: {info}"
        raise NotEnoughInfoError(msg)

    search_results: dict[SongInfo, APIResultList[SongInfo]] = {}
    songs_score: dict[SongInfo, float] = {}
    lyrics_results: dict[SongInfo, Lyrics] = {}
    errors: list[Exception] = []

    with ThreadPoolExecutor() as executor:
        search_tasks: list[Future] = []
        
        # Initial search
        keyword_to_search = keywords.get("artist-title") or keywords.get("title") or keywords["file_name"]
        for source in sources:
            future = executor.submit(search, source, keyword_to_search, SearchType.SONG)
            search_tasks.append(future)

        # Wait for initial search results
        completed_searches, _ = wait(search_tasks, timeout=timeout, return_when=ALL_COMPLETED)

        potential_lyrics_tasks: dict[Future, SongInfo] = {}

        for future in completed_searches:
            try:
                results: APIResultList[SongInfo] = future.result()
                if not results or not isinstance(results.info, SearchInfo):
                    continue

                result_score: list[tuple[float, SongInfo]] = []
                for result in results:
                    if info.duration and abs((info.duration or -4) - (result.duration or -8)) > 4000:
                        continue
                    
                    if results.info.keyword in (keywords.get("artist-title"), keywords.get("title")):
                        title_score = calculate_title_score(info.title or "", result.title or "")
                        album_score = max(text_difference(info.album.lower(), result.album.lower()) * 100, 0) if info.album and result.album else None
                        artist_score = calculate_artist_score(str(info.artist), str(result.artist)) if info.artist and result.artist else None
                        score = title_score
                        if artist_score is not None:
                            score = max(title_score * 0.5 + artist_score * 0.5, (title_score * 0.5 + artist_score * 0.35 + (album_score or 0) * 0.15) if album_score is not None else 0)
                        elif album_score:
                            score = max(title_score * 0.7 + album_score * 0.3, title_score * 0.8)
                        if title_score < 30:
                            score = max(0, score - 35)
                    else:
                        score = max(text_difference(keywords["file_name"], result.title or "") * 100, text_difference(keywords["file_name"], f"{result.artist!s} - {result.title}")*100)

                    if score > min_score:
                        result_score.append((score, result))

                result_score.sort(key=lambda x: x[0], reverse=True)

                # Submit tasks to get lyrics for top candidates
                for i, (score, song_candidate) in enumerate(result_score):
                    if i >= 2: break # Try top 2 candidates
                    songs_score[song_candidate] = score
                    search_results[song_candidate] = APIResultList([song_candidate, *[r for r in results if r != song_candidate]], results.info)
                    task = executor.submit(get_lyrics, song_candidate)
                    potential_lyrics_tasks[task] = song_candidate

            except Exception as e:
                errors.append(e)

        # Wait for lyrics results
        for future in as_completed(potential_lyrics_tasks):
            song_info_candidate = potential_lyrics_tasks[future]
            try:
                lyrics = future.result()
                if lyrics:
                    lyrics_results[song_info_candidate] = lyrics
            except Exception as e:
                errors.append(e)

    if not lyrics_results:
        if any(not isinstance(e, LyricsNotFoundError) for e in errors):
             logger.error(f"Errors during auto_fetch: {errors}")
        raise LyricsNotFoundError("没有找到符合要求的歌曲")

    highest_score = max(songs_score.get(song_info, 0) for song_info in lyrics_results)
    lyrics_results = {
        song_info: lyrics
        for song_info, lyrics in lyrics_results.items()
        if abs(songs_score.get(song_info, 0) - highest_score) <= 15
    }

    def get_rank(lyrics: Lyrics) -> int:
        rank = 0
        if lyrics.types.get("orig") == LyricsType.VERBATIM: rank += 10
        if "ts" in lyrics: rank += 5
        if "roma" in lyrics: rank += 2
        return rank

    sorted_lyrics = sorted(lyrics_results.items(), key=lambda item: get_rank(item[1]), reverse=True)
    
    final_lyrics_list = [item[1] for item in sorted_lyrics]

    for source_priority in sources:
        for lyrics in final_lyrics_list:
            if lyrics.info.source == source_priority:
                if not return_search_results:
                    return lyrics
                
                info_key = next(s_info for s_info, l in lyrics_results.items() if l == lyrics)
                
                all_search_results = reduce(lambda a, b: a + b, search_results.values()) if search_results else APIResultList([])
                
                return lyrics, APIResultList(search_results.get(info_key, APIResultList([])) + all_search_results)

    # Fallback if no priority source matched
    best_lyrics, all_results = sorted_lyrics[0][1], reduce(lambda a, b: a + b, search_results.values(), APIResultList([]))
    if return_search_results:
        return best_lyrics, all_results
    return best_lyrics 