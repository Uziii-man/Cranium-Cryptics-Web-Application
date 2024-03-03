// Function to handle file input change event
function handleFileInputChange(event) {
    const file = event.target.files[0]; // Get the selected file
    if (file) {
        // Create a FileReader object
        const reader = new FileReader();
        reader.onload = function(e) {
            // Set the src attribute of the preview image to the data URL
            document.getElementById('previewImage').src = e.target.result;
        }
        // Read the file as a data URL (base64 encoded)
        reader.readAsDataURL(file);
    }
}

// Add event listener for file input change
document.getElementById('uploadInput').addEventListener('change', handleFileInputChange);

function clearFileInput() {
    updateUserName.innerHTML = userData.userName;
    const userControl = document.querySelector('.user-control');
    const emailControl = document.querySelector('.email-control');
    const passwordControl = document.querySelector('#change__password');
    const confirmPasswordControl = document.querySelector('#change__password__confirm');
    const userName = document.querySelector('.user-name');

    userControl.value = userData.userName;
    emailControl.value = userData.email;
    passwordControl.value = userData.password;
    confirmPasswordControl.value = userData.password;
    userName.innerHTML = userData.userName;
}

document.getElementById("profileForm").addEventListener('submit', updateUserDetails);

function updateUserDetails(e) {
  e.preventDefault();

  const firebaseConfig = {
          apiKey: "AIzaSyChuCsLIaZDj4nfI1jE9Dpbkt2CFZSKR1c",
          authDomain: "craniumcryptics.firebaseapp.com",
          databaseURL: "https://craniumcryptics-default-rtdb.firebaseio.com",
          projectId: "craniumcryptics",
          storageBucket: "craniumcryptics.appspot.com",
          messagingSenderId: "415799673124",
          appId: "1:415799673124:web:31e4bf310fda12df4c73b1"
  };

  // Initialize Firebase if it's not already initialized
  if (!firebase.apps.length) {
    firebase.initializeApp(firebaseConfig);
  }

  // Get a reference to the default Firebase app
  const defaultApp = firebase.app();

  const database = firebase.database();

  const accountName = userData.userName;

  const formUserName = document.querySelector('#change__userName').value;
  const formEmail = document.querySelector('#change__email').value;
  const formPassword = document.querySelector('#change__password').value;
  const formConfirmPassword = document.querySelector('#change__password__confirm').value;

  // Checking input field validations
  if (!isValidEmail(formEmail)) {
    alert("Please Enter A Valid Email Address");
    return;
  }

  if (!isValidPassword(formPassword)) {
    alert("Password Should Be 6 Characters Long With 1 Symbol And 1 Capital Letter.");
    return;
  }

  if (formPassword !== formConfirmPassword) {
    alert("Password And Confirm Password Do Not Match.");
    return;
  }

  if (!isValidUsername(formUserName)) {
    alert("Username Should Be 8 Characters Long");
    return;
  }

  const entryRef = database.ref('UserData/' + formUserName);

  if(accountName === formUserName){
       const updates = {
          confirmPassword: formConfirmPassword,
          email: formEmail,
          password: formPassword,
          userName: formUserName,
        };
      firebase.database().ref('UserData/' + accountName).update(updates);
      alert("User details updated successfully");
  }else{

      entryRef.once('value')
    .then(snapshot => {
      if (snapshot.exists()) {
        alert("Username already exists. Please choose a different username.");
      } else {
        // Username doesn't exist, update the user details
        const updates = {
          confirmPassword: formConfirmPassword,
          email: formEmail,
          password: formPassword,
          userName: formUserName,
        };


       const entryRef = database.ref('UserData/' + accountName);
       entryRef.remove()
       firebase.database().ref('UserData/' + formUserName).update(updates);


        alert("User details updated successfully");
      }
    })
    .catch(error => {
      console.error("Error checking username existence:", error);
      alert("An error occurred. Please try again later.");
    });

  }

   userData.userName = formUserName;
   userData.email = formEmail;
   userData.password = formPassword;
   userData.confirmPassword = formConfirmPassword;

   localStorage.setItem('userData', JSON.stringify(userData));

   clearFileInput();

}


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

function isValidUsername(username) {
  if (username.length > 8) {
      return false;
  }

  var usernameRegex = /^[a-zA-Z0-9_@]*[a-zA-Z0-9]+[a-zA-Z0-9_@]*$/;
  return usernameRegex.test(username);
}

// function doesUsernameExist(username) {
//     alert(username)
//
//     if (username === userData.userName) {
//         alert(123)
//         return true;
//     }else{
//         return new Promise((resolve, reject) => {
//             hForm.child(username).once('value', (snapshot) => {
//                 resolve(snapshot.exists());
//             });
//         });
//     }
// }


function isValidEmail(email){
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











