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
        
    fetch('https://randomuser.me/api/')
    .then(response => {
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      return response.json(); 
    })
    .then(data => {
      console.log(data);
    })
    .catch(error => {
      console.error('There was a problem with the API request:', error);
    });
  }
}
