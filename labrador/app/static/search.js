
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

document.getElementById('batch-button').addEventListener('click', async (event) => {
    event.preventDefault();

    const query = document.getElementById('search-input').value;
    instructionsContainer.style.display = 'none';
    searchContainer.style.display = 'none';
    showLoadingModal();

    let uid = Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
    await requestBatch(uid, query);
});

async function requestBatch(uid, query) {
    try {
        await Promise.all([
            receiveData(uid, query, `/missing?uid=${uid}`),
        ]);
    } catch (error) {
        console.error("Error in making requests:", error);
        // Handle the error appropriately
    }
}

async function makeRequests(uid, query) {
    try {
        await Promise.all([
            receiveData(uid, query, `/keyword_search?query=${query}&uid=${uid}`),
            receiveData(uid, query, `/neural_search?query=${query}&uid=${uid}`),
            receiveData(uid, query, `/repository_search?query=${query}&uid=${uid}`)
        ]);
    } catch (error) {
        console.error("Error in making requests:", error);
        // Handle the error appropriately
    }
}

async function receiveData(uid, query, responseInput) {
    fetch(responseInput)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            processData(data);
        })
        .catch(error => {
            console.error("Error in response:", error);
        });
}

// DataStreaming
//while (true) {
//         const { value, done } = await reader.read();
//         const chunk = new TextDecoder().decode(value);
//         consumedData += chunk;
//         let j = consumedData.indexOf('\n', i);
//         if (j === -1) {
//             continue;
//         }
//         let nextJsonString= consumedData.slice(i, j);
//         i = j + 1;
//         try {
//             const data = JSON.parse(nextJsonString);
//             if (data.success) {
//                 return processData(data);
//             } else {
//                 console.error("Error in data:", data);
//                 break;
//             }
//         } catch (e) {
//             if (consumedData === "") {
//                 throw e;
//             }
//         }
//     }
//     await sleep(100)
// const reader = await Promise.race([responsePromise, timeout(TIMEOUT_MS, "Request timed out")]);
// let consumedData = "";
// let i = 0;


function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function processData(data) {
    data.hits.forEach((item) => {
        const div = document.createElement('div');
        div.classList.add('box');
        div.id = `${item.doc_id}`;
        div.style.display = 'block';
        div.innerHTML = ''
        if (item.query && item.query !== '') {
            div.innerHTML += `<p id="query" ><strong style="color: red;">Consulta:</strong> <i>${item.query}</i></p>`
        }
        div.innerHTML += `
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
        documentDivPool.push([item.query_id, div]);
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
    queryId = currentDisplayedDocument[0];
    document = currentDisplayedDocument[1];
    documentContainer.appendChild(currentDisplayedDocument[1]);
    documentDivPool.splice(randomIndex, 1);

    labeledDocuments++;
}

document.addEventListener('click', async (event) => {
    if (event.target.matches('#annotate-button')) {
        const docId = currentDisplayedDocument[1].id;
        const selectedLabel = document.querySelector(`input[name="label-${docId}"]:checked`).value;
        try {
            const response = await fetch(`/annotate?query_id=${queryId}&doc_id=${docId}&rel=${selectedLabel}`);
            const data = await response.json();
            if (data.success) {
                documentContainer.removeChild(documentContainer.lastChild);
                currentDisplayedDocument[1].style.display = 'none';
                displayNextDocument();
            } else {
                alert('Error submitting label. Please check the console for more details.');
            }
        } catch (error) {
            console.error("There was an error:", error);
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
