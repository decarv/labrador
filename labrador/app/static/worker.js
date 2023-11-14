
self.addEventListener('message', async (event) => {
    const query = event.data.query;
    await streamData(query);
});

async function streamData(query) {
    const response = await fetch(`/search?query=${query}`);
    const reader = response.body.getReader();
    let consumedData = "";
    let i = 0;

    while (true) {
        const { value, done } = await reader.read();
        if (done) {
            break;
        }
        const chunk = new TextDecoder().decode(value);
        consumedData += chunk;
        let j = consumedData.indexOf('\n', i);
        if (j === -1) {
            continue;
        }
        let nextJsonString = consumedData.slice(i, j);
        i = j + 1;
        try {
            const data = JSON.parse(nextJsonString);
            if (data.success) {
                self.postMessage({ type: 'data', data: data });
                await sleep(50);
                if (data.done) {
                    break;
                }
            } else {
                console.error("Error in data:", data);
                break;
            }
        } catch (e) {
            console.error("Error parsing JSON:", e);
        }
    }
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
