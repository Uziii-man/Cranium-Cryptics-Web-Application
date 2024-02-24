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


document.getElementById("loginForm").addEventListener('submit', submitLoginForm);

function submitLoginForm(e) {
    e.preventDefault();

    var loginUsername = getElementVal('login__username');
    var loginPassword = getElementVal('login__password');

    viewData(loginUsername, loginPassword)

}

const viewData = (loginUsername, loginPassword) => {

    hForm.child(loginUsername).once('value', function(snapshot) {
        var userData = snapshot.val();
        if (userData) {
            var storedPassword = userData.password;
            if (loginPassword === storedPassword) {
              window.location.href = '/Dashboard'
              console.log("Login successful");
              
            } else {
              console.log("Incorrect password");
              alert("Incorrect password")
            }
          } else {
            console.log("User not found");
            alert("User not Found")
          }
    });

}






