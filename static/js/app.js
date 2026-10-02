/**
 * Spotify Semantic Retrieval - Editorial Client-side Search
 */

document.addEventListener('DOMContentLoaded', () => {
    const searchInput = document.getElementById('search-input');
    const searchBtn = document.getElementById('search-btn');
    const topKSelect = document.getElementById('top-k-select');
    const vibeChips = document.querySelectorAll('.vibe-chip');
    const latencyVal = document.getElementById('latency-val');

    const initialState = document.getElementById('initial-state');
    const loadingState = document.getElementById('loading-state');
    const noResultsState = document.getElementById('no-results-state');
    const resultsGrid = document.getElementById('results-grid');

    let debounceTimer = null;

    // -------------------------------------------------------------------------
    // Perform Search
    // -------------------------------------------------------------------------
    async function performSearch(query) {
        const cleanQuery = query.trim();
        if (!cleanQuery) {
            showState('initial');
            latencyVal.textContent = 'LATENCY: 0.0 MS';
            return;
        }

        const topK = topKSelect.value || 10;
        showState('loading');

        try {
            const url = `/api/search?q=${encodeURIComponent(cleanQuery)}&top_k=${topK}`;
            const response = await fetch(url);

            if (!response.ok) {
                throw new Error(`Server status: ${response.status}`);
            }

            const data = await response.json();

            // Update header latency
            if (data.latency_ms !== undefined) {
                latencyVal.textContent = `LATENCY: ${data.latency_ms} MS`;
            }

            if (!data.results || data.results.length === 0) {
                showState('no-results');
            } else {
                renderResults(data.results);
                showState('results');
            }

        } catch (error) {
            console.error('Search error:', error);
            showState('no-results');
        }
    }

    // -------------------------------------------------------------------------
    // Render Results in Minimalist Editorial List
    // -------------------------------------------------------------------------
    function renderResults(songs) {
        resultsGrid.innerHTML = '';

        songs.forEach(song => {
            const item = document.createElement('div');
            item.className = 'result-item';

            // Format rank like [01], [02], [10]
            const rankFormatted = song.rank < 10 ? `0${song.rank}` : `${song.rank}`;

            item.innerHTML = `
                <div class="item-top">
                    <div class="item-title-group">
                        <span class="rank-tag">[${rankFormatted}]</span>
                        <span class="song-name">${escapeHtml(song.song)}</span>
                        <span class="title-sep">•</span>
                        <span class="artist-name">${escapeHtml(song.artist)}</span>
                    </div>
                    <span class="score-tag">${song.match_percentage}</span>
                </div>
                <div class="lyrics-block">
                    "${escapeHtml(song.lyrics_snippet)}..."
                </div>
                <div class="item-bottom">
                    <a href="${song.spotify_url}" target="_blank" rel="noopener noreferrer" class="spotify-link">
                        Play on Spotify ↗
                    </a>
                </div>
            `;

            resultsGrid.appendChild(item);
        });
    }

    // -------------------------------------------------------------------------
    // State Manager
    // -------------------------------------------------------------------------
    function showState(state) {
        initialState.classList.add('hidden');
        loadingState.classList.add('hidden');
        noResultsState.classList.add('hidden');
        resultsGrid.classList.add('hidden');

        if (state === 'initial') {
            initialState.classList.remove('hidden');
        } else if (state === 'loading') {
            loadingState.classList.remove('hidden');
        } else if (state === 'no-results') {
            noResultsState.classList.remove('hidden');
        } else if (state === 'results') {
            resultsGrid.classList.remove('hidden');
        }
    }

    function escapeHtml(text) {
        if (!text) return '';
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, m => map[m]);
    }

    // -------------------------------------------------------------------------
    // Event Listeners
    // -------------------------------------------------------------------------
    searchBtn.addEventListener('click', () => {
        performSearch(searchInput.value);
    });

    searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            performSearch(searchInput.value);
        }
    });

    searchInput.addEventListener('input', () => {
        const val = searchInput.value;
        clearTimeout(debounceTimer);
        if (val.trim().length >= 3) {
            debounceTimer = setTimeout(() => {
                performSearch(val);
            }, 450);
        }
    });

    topKSelect.addEventListener('change', () => {
        if (searchInput.value.trim()) {
            performSearch(searchInput.value);
        }
    });

    vibeChips.forEach(chip => {
        chip.addEventListener('click', () => {
            const vibe = chip.getAttribute('data-vibe');
            searchInput.value = vibe;
            performSearch(vibe);
        });
    });
});
