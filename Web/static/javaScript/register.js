// Firebase Configurations
const firebaseConfig = {
  apiKey: "AIzaSyChuCsLIaZDj4nfI1jE9Dpbkt2CFZSKR1c",
  authDomain: "craniumcryptics.firebaseapp.com",
  databaseURL: "https://craniumcryptics-default-rtdb.firebaseio.com",
  projectId: "craniumcryptics",
  storageBucket: "craniumcryptics.appspot.com",
  messagingSenderId: "415799673124",
  appId: "1:415799673124:web:31e4bf310fda12df4c73b1"
};

// Initialize Firebase
firebase.initializeApp(firebaseConfig);

// Get Element By ID
const getElementVal = (id) => {
  return document.getElementById(id).value;
};

var hForm = firebase.database().ref('UserData');

document.getElementById("regForm").addEventListener('submit', submitForm);

// Submit Form Function
function submitForm(e) {
  e.preventDefault();

  var email = getElementVal('reg_email');
  var userName = getElementVal('reg_username');
  var password = getElementVal('reg_password');
  var confirmPassword = getElementVal('reg_confirm');

  // Checking input field validations
  // If any of the validation fails, the function will return and the form will not be submitted

  // Email Validation
  if (!isValidEmail(email)) {
    displayAlert("Please Enter A Valid Email Address");
    return;
  }

  // Password Validation
  if (!isValidPassword(password)) {
    displayAlert("Password Should Be at least 6 Characters Long With 1 Symbol And 1 Capital Letter.");
    return;
  }

  // Confirm Password Validation
  if (password !== confirmPassword) {
    displayAlert("Password And Confirm Password Do Not Match.");
    return;
  }

  // Username Validation
  if(!isValidUsername(userName)){
    displayAlert("Username Should Be 8 Characters Long");
    return;
  }

  // Checking if the username already exists
  doesUsernameExist(userName)
    .then(usernameExists => {
      if (usernameExists) {
        displayAlert("Username Already Exists. Enter A Different One. Try Using @ Or Underscores.");
        return;
      }

  // If all the validations pass, the data will be stored in the firebase DB
  saveMessages(email, userName, password, confirmPassword);
  displayAlert("Registration Successfull");
  setTimeout(() => {
  window.location.href = '/login';
    }, 2000);
  });
}

// Storing Data to firebase DB
const saveMessages = (email, userName, password, confirmPassword) => {
  var newhForm = hForm.child(userName);

  newhForm.set({
    email: email,
    userName: userName,
    password: password,
    confirmPassword: confirmPassword
  });
};

// Validating Email
function isValidEmail(email) {
  var domain = email.split('@')[1];
  var recognizableProviders = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com"];

  if (domain.toLowerCase().includes(".lk") || domain.toLowerCase().includes(".gov")) {
      return true;
  }

  if (!recognizableProviders.includes(domain.toLowerCase())) {
      return false;
  }

  var emailRegex = /^[^\s@]+@(?:[^\s@]+\.)+[^\s@]+$/;
  return emailRegex.test(email);
}

//Duplication check for username
function doesUsernameExist(username) {
  return new Promise((resolve, reject) => {
    hForm.child(username).once('value', (snapshot) => {
      resolve(snapshot.exists());
    });
  });
}

// Validating Username
function isValidUsername(username) {
  if (username.length < 8) {
      return false;
  }

  var usernameRegex = /^[a-zA-Z0-9_@]*[a-zA-Z0-9]+[a-zA-Z0-9_@]*$/;
  return usernameRegex.test(username);
}

//Validating Password
function isValidPassword(password) {

    if (password.length < 6) {
        return false;
    }

    var symbolRegex = /[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/;
    if (!symbolRegex.test(password)) {
        return false;
    }

    var capitalLetterRegex = /[A-Z]/;
    if (!capitalLetterRegex.test(password)) {
        return false;
    }
    return true;
}

// Customized alert messages
function displayAlert(message) {
  var issue = document.getElementById("customAlert");
  var alertMessage = document.getElementById("alertMessage");
  alertMessage.innerHTML = message;
  issue.style.display = "block";

  // Closes the alert popup when anywhere in the screen is clicked
  window.onclick = function (event) {
    if (event.target == issue) {
      issue.style.display = "none";
    }
  };
}


