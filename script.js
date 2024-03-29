function readTextFromClass(className) {
  let elements = document.querySelectorAll("." + className);
  let allText = "";
  elements.forEach(function (element) {
    allText += element.innerText.trim() + "\n";
  });
  console.log(allText);
}

chrome.runtime.onMessage.addListener(function (message, sender, sendResponse) {
  if (message.action === "readText") {
    readTextFromClass("XbIp4");
  }
});
