
const instructionsContainer = document.getElementById('instructions-container');
const searchContainer = document.getElementById('search-container');
const documentContainer = document.getElementById('document-container');

let documentDivPool = []
let started = false;
let labeledDocuments = 0;
let confirmButton = null;
let queryId = null;
let currentDisplayedDocument = null;


document.getElementById('search-form').addEventListener('submit', async (event) => {
    event.preventDefault();

    const query = document.getElementById('search-input').value;
    if (query === '') {
        alert('Por favor, use a caixa de consulta.');
        return;
    }
    instructionsContainer.style.display = 'none';
    searchContainer.style.display = 'none';
    showLoadingModal();

    let uid = Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
    await makeRequests(uid, query);
});

async function makeRequests(uid, query) {
    await receiveData(uid, query, `/neural_search?query=${query}&uid=${uid}`);
    await receiveData(uid, query, `/repository_search?query=${query}&uid=${uid}`);
}

function timeout(ms, error) {
    return new Promise((_, reject) => setTimeout(() => reject(new Error(error)), ms));
}

async function receiveData(uid, query, responseInput) {
    const TIMEOUT_MS = 20000; // 20 seconds
    const responsePromise = fetch(responseInput).then(response => response.body.getReader());
    const reader = await Promise.race([responsePromise, timeout(TIMEOUT_MS, "Request timed out")]);
    let consumedData = "";
    let i = 0;

    while (true) {
        const { value, done } = await reader.read();
        const chunk = new TextDecoder().decode(value);
        consumedData += chunk;
        let j = consumedData.indexOf('\n', i);
        if (j === -1) {
            continue;
        }
        let nextJsonString= consumedData.slice(i, j);
        i = j + 1;
        try {
            const data = JSON.parse(nextJsonString);
            if (data.success) {
                processData(data);
                return;
            } else {
                console.error("Error in data:", data);
                break;
            }
        } catch (e) {
            if (consumedData === "") {
                throw e;
            }
        }
        await sleep(10);
    }
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function processData(data) {
    queryId = data.queryId;
    data.hits.forEach((item) => {
        const div = document.createElement('div');
        div.classList.add('box');
        div.id = `${item.doc_id}`;
        div.style.display = 'block';
        div.innerHTML = `
    <h2 id="title">${item.title}</h2>
    <p id="author"><small>${item.author}</small></p>
    <p id="abstract">${item.abstract}</p>
    <p id="keywords">${item.keywords}</p>
    <div id="labels">
        <div class="custom-radio">
            <input type="radio" id="label1-${item.doc_id}" name="label-${item.doc_id}" value="1">
            <label for="label1-${item.doc_id}">1</label>
        </div>
        <div class="custom-radio">
            <input type="radio" id="label2-${item.doc_id}" name="label-${item.doc_id}" value="2">
            <label for="label2-${item.doc_id}">2</label>
        </div>
        <div class="custom-radio">
            <input type="radio" id="label3-${item.doc_id}" name="label-${item.doc_id}" value="3">
            <label for="label3-${item.doc_id}">3</label>
        </div>
        <div class="custom-radio">
            <input type="radio" id="label4-${item.doc_id}" name="label-${item.doc_id}" value="4">
            <label for="label4-${item.doc_id}">4</label>
        </div>
        <div class="custom-radio">
            <input type="radio" id="label5-${item.doc_id}" name="label-${item.doc_id}" value="5">
            <label for="label5-${item.doc_id}">5</label>
        </div>
        <button id="annotate-button">Confirmar</button>
    </div>
    `;
        documentDivPool.push(div);
    });

    // If it is the first document to be labeled
    if (labeledDocuments === 0) {
        hideLoadingModal();
        documentContainer.style.display = 'block';
        displayNextDocument();
    }
}

function displayNextDocument() {
    if (documentDivPool.length === 0) {
        showThankYouModal();
        return;
    }
    let randomIndex = Math.floor(Math.random() * documentDivPool.length);
    currentDisplayedDocument = documentDivPool[randomIndex];
    documentContainer.appendChild(currentDisplayedDocument);
    documentDivPool.splice(randomIndex, 1);

    labeledDocuments++;
}

document.addEventListener('click', async (event) => {
    if (event.target.matches('#annotate-button')) {
        const docId = currentDisplayedDocument.id;
        const selectedLabel = document.querySelector(`input[name="label-${docId}"]:checked`).value;
        try {
            const response = await fetch(`/annotate?query_id=${queryId}&doc_id=${docId}&rel=${selectedLabel}`);
            const data = await response.json();
            if (data.success) {
                documentContainer.removeChild(documentContainer.lastChild);
                currentDisplayedDocument.style.display = 'none';
                displayNextDocument();
            } else {
                alert('Error submitting label. Please try again.');
            }
        } catch (error) {
            console.error("There was an error:", error);
            alert('Error submitting label. Please check the console for more details.');
        }
    }
})

function showLoadingModal() {
    document.getElementById('loadingModal').style.display = "block";
}

function hideLoadingModal() {
    if (document.getElementById('loadingModal').style.display === "block") {
        document.getElementById('loadingModal').style.display = "none";
    }
}

function showThankYouModal() {
    const modal = document.getElementById('thankyou-modal');
    modal.style.display = 'block';
}

// // To hide the modal and redirect to the search page
// document.getElementById('redirect-button').addEventListener('click', function() {
//     const modal = document.getElementById('thankyou-modal');
//     modal.style.display = 'none';
//     window.location.href = '/';  // Assuming the search page is the root URL
// });

// Display the modal when the page loads
window.onload = function() {
    const modal = document.getElementById('disclaimer-modal');
    modal.style.display = 'block';
}

// Hide the modal when the "Accept" button is clicked
document.getElementById('accept-button').addEventListener('click', function() {
    const modal = document.getElementById('disclaimer-modal');
    modal.style.display = 'none';
});
