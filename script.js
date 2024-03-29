chrome.runtime.onMessage.addListener(function (message, sender, sendResponse) {
  if (message.action === "readText") {
    readTextFromClass("XbIp4"); //outlook
    readTextFromClass("a3s"); //gmail
  }
});

function readTextFromClass(className) {
  var loadingDiv = document.getElementById("loading");
  let elements = document.querySelectorAll("." + className);
  let allText = "";
  elements.forEach(function (element) {
    allText += element.innerText.trim() + "\n";
  });
  // Add api to send to model
  console.log(allText);
  if (loadingDiv) {
    loadingDiv.style.display = "block";
    fakeAsyncFunction()
      .then(function (percentage) {
        console.log("Percentage:", percentage);
        loadingDiv.style.display = "none";
      })
      .catch(function (error) {
        console.error("Error:", error);
        loadingDiv.style.display = "none";
      });
  }
}

function fakeAsyncFunction() {
  return new Promise(function (resolve, reject) {
    setTimeout(function () {
      var percentage = 50;
      resolve(percentage);
    }, 4000);
  });
}
