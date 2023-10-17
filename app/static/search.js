document.getElementById('search-form').addEventListener('submit', async (event) => {
    event.preventDefault();

    const query = document.getElementById('search-input').value;
    const data = await fetchData(query);
    displayData(data);
});

async function fetchData(query) {
    const response = await fetch(`/search?query=${query}`);
    const data = await response.json();
    return data;
}

let currentCardId = 0;  // Track the currently displayed card

function displayData(data) {
    const searchContainer = document.getElementById('search-container');
    searchContainer.innerHTML = '';
    const container = document.getElementById('data-container');
    container.innerHTML = '';
    
    data.hits.forEach((item, index) => {
        const div = document.createElement('div');
        div.classList.add('qrel-card');
        div.id = `card-${index}`;
        div.style.display = (index === 0) ? 'block' : 'none';  // Only show the first card initially
        div.innerHTML = `
            <h2 id="title">${item.title}</h2>
            <p id="author"><small>${item.author}</small></p>
            <p id="abstract">Resumo: ${item.abstract}</p>
            <p id="keywords">Palavras-chave: ${item.keywords}</p>
            <div id="labels">
                <label><input type="radio" name="label-${item.doc_id}" value="1"> 1</label>
                <label><input type="radio" name="label-${item.doc_id}" value="2"> 2</label>
                <label><input type="radio" name="label-${item.doc_id}" value="3"> 3</label>
                <label><input type="radio" name="label-${item.doc_id}" value="4"> 4</label>
                <label><input type="radio" name="label-${item.doc_id}" value="5"> 5</label>
            </div>
            <button onclick="submitLabelAndNext(${data.queryId}, ${item.doc_id}, ${index})">Confirmar</button>
        `;
        container.appendChild(div);
    });
}

async function submitLabelAndNext(queryId, itemId, cardIndex) {
    const selectedLabel = document.querySelector(`input[name="label-${itemId}"]:checked`).value;

    try {
        const response = await fetch(`/annotate/?query_id=${queryId}&doc_id=${itemId}&rel=${selectedLabel}`);
        const data = await response.json();

        if (data.success) {
            // Hide current card
            document.getElementById(`card-${cardIndex}`).style.display = 'none';

            // Show the next card
            currentCardId++;
            const nextCard = document.getElementById(`card-${currentCardId}`);
            if (nextCard) {
                nextCard.style.display = 'block';
            } else {
                alert('All qrels have been labeled!');
            }
        } else {
            alert('Error submitting label. Please try again.');
        }
    } catch (error) {
        console.error("There was an error:", error);
        alert('Error submitting label. Please check the console for more details.');
    }
}
