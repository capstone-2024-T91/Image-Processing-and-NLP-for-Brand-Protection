chrome.runtime.onMessage.addListener(function (message, sender, sendResponse) {
  if (message.action === "readText") {
    chrome.scripting.executeScript({
      target: { tabId: sender.tab.id },
      function: readTextFromPage,
      args: [message.className],
    });
  }
});

function readTextFromPage(className) {
  let elements = document.querySelectorAll("." + className);
  let allText = "";
  elements.forEach(function (element) {
    allText += element.innerText.trim() + "\n";
  });
  console.log(allText);
}
