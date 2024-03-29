document.addEventListener("DOMContentLoaded", function () {
  var readButton = document.getElementById("readButton");
  readButton.addEventListener("click", function () {
    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
      chrome.tabs.sendMessage(tabs[0].id, { action: "readText" });
    });
  });
});
