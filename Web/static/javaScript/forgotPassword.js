// Firebase configuration
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

var hForm = firebase.database().ref('UserData')


// Event listener for form submission
document.getElementById('regForm').addEventListener('submit', function (event) {
    event.preventDefault();

    var email = document.getElementById('reg_email').value;
    var username = document.getElementById('reg_username').value;
    var password = document.getElementById('reg_password').value;
    var confirmPassword = document.getElementById('reg_confirm').value;

    viewData(username, email, password, confirmPassword);
});

// Function to retrieve user data from the database
async function viewData(username, email, password, confirmPassword) {
    try {
        const snapshot = await firebase.database().ref('UserData/' + username).once('value');
        const userData = snapshot.val();

        if (!userData) {
            alert('User data not found');
            return;
        }

        const storedUserName = userData.userName;
        const storedEmail = userData.email;

        if (validateEmail(email, storedEmail) && validateUsername(username, storedUserName)) {
            if (password === confirmPassword) {
                const updates = {
                    confirmPassword: confirmPassword,
                    email: email,
                    password: password,
                    userName: username,
                };

                if(!isValidPassword(password)){
                    alert("Password Should Be  6 Characters Long With 1 Symbol And 1 Capital Letter.");
                    return;
                }

                await firebase.database().ref('UserData/' + username).update(updates);
                alert('Password changed successfully!');
                window.location.href = '/login';
            } else {
                alert('Passwords do not match');
            }
        } else {
            alert('Invalid email or username');
        }
    } catch (error) {
        console.error('Error updating user data:', error);
        alert('An error occurred. Please try again later.');
    }
}


// Email validation function
function validateEmail(given_email, email) {
    return given_email === email;
}


// Username validation function
function validateUsername(given_username, userName) {
    return given_username === userName;
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
