
const firebaseConfig = {
  apiKey: "AIzaSyChuCsLIaZDj4nfI1jE9Dpbkt2CFZSKR1c",
  authDomain: "craniumcryptics.firebaseapp.com",
  databaseURL: "https://craniumcryptics-default-rtdb.firebaseio.com",
  projectId: "craniumcryptics",
  storageBucket: "craniumcryptics.appspot.com",
  messagingSenderId: "415799673124",
  appId: "1:415799673124:web:31e4bf310fda12df4c73b1"
};

firebase.initializeApp(firebaseConfig);

const getElementVal = (id) => {
return document.getElementById(id).value;
};

var hForm = firebase.database().ref('UserData')

document.getElementById("regForm").addEventListener('submit', submitForm)

function submitForm(e){
e.preventDefault();

var email = getElementVal('reg_email')
var userName = getElementVal('reg_username')
var password = getElementVal('reg_password')
var confirmPassword = getElementVal('reg_confirm')

// Checking inout field validations
if (!isValidEmail(email)) {
  displayAlert("Please Enter A Valid Email Address");
  return;
}


if (!isValidPassword(password)) {
  displayAlert("Password Should Be At Least 6 Characters Long.");
  return;
}

if (password !== confirmPassword) {
  displayAlert("Password and confirm password do not match.");
  return;
}

saveMessages(email,userName,password,confirmPassword)
displayAlert("Registration Successfull")

}

// Storing Data to firebase DB
const saveMessages = (email, userName, password, confirmPassword) => {
var newhForm = hForm.child(userName);

newhForm.set({
  email : email,
  userName : userName,
  password : password,
  confirmPassword : confirmPassword,

  })

};

// Validating Email
function isValidEmail(email) {
  var emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

//Should implement duplication check for username

// VAlidating password
function isValidPassword(password) {
  
  return password.length >= 6;
}


// Customized alert messages
function displayAlert(message) {
  var issue = document.getElementById("customAlert");
  var alertMessage = document.getElementById("alertMessage");
  alertMessage.innerHTML = message;
  issue.style.display = "block";
  

  //Closes the alert popup when anywhere in the screen is clicked
  window.onclick = function(event) {
    if (event.target == issue) {
      issue.style.display = "none";
    }
  }
  
}