chrome.runtime.onMessage.addListener(function (message, sender, sendResponse) {
  if (message.action === "readText") {
    readTextFromClass("XbIp4"); //outlook
    readTextFromClass("a3s"); //gmail
  }
});

function readTextFromClass(className) {
  let elements = document.querySelectorAll("." + className);
  let allText = "";
  elements.forEach(function (element) {
    allText += element.innerText.trim() + "\n";
  });
  // Add api to send to model
  console.log(allText);
}
