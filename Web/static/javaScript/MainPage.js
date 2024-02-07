let pop = document.querySelector(".profile-popup");

function toggleMenu() {
    const menuContainer = document.querySelector('.menu-container');
    const leftPane = document.querySelector('.left-pane');
    const mainMenu = document.querySelector('.main-menu');
    const menuButton = document.querySelector('.menu-button');
    const menuBurger = document.querySelector('.menu-burger');
    const diseaseCard = document.querySelector('.image-detection-cards');
    const card1 = document.querySelector('.card-1');
    const card2 = document.querySelector('.card-2');
    const card3 = document.querySelector('.card-3');

    if (menuContainer.style.display === 'none' || menuContainer.style.display === '') {
        menuContainer.style.display = 'block';
        leftPane.style.width = '250px';
        mainMenu.style.display = 'block';
        menuButton.display = 'flex';
        menuButton.flexDirection = 'column';
        menuBurger.style.margin_right = '20px';
        menuBurger.innerHTML = '&times;';
        menuBurger.style.fontSize = '35px';
        menuBurger.style.paddig_right = '20px';
        leftPane.style.Color= "#FFF";

    } else {
        menuContainer.style.display = 'none';
        leftPane.style.width = '40px';
        mainMenu.style.display = 'none';
        menuBurger.innerHTML = '&#9776;';
        menuBurger.style.padding_right = '0px';
        menuBurger.style.fontSize = '30px';
        menuBurger.classList.remove('rotate');
        leftPane.style.Color = "#000";
    }
}


menuBurger.addEventListener('', function() {
    menuBurger.classList.add('rotate'); 
});

menuBurger.addEventListener('mouseleave', function() {
    menuBurger.classList.remove('rotate'); 
});



function ProfilePopView(){
   pop.style.display = "flex";
}

function removeProfileView(){
    pop.style.display = "none"; 
}

document.addEventListener('DOMContentLoaded', function () {
    const fileInput = document.getElementById('fileInput');
    const imageDisplay = document.getElementById('imageDisplay');

    fileInput.addEventListener('change', function (event) {
        const file = event.target.files[0];

        if (file) {
            const reader = new FileReader();

            reader.onload = function (e) {
                imageDisplay.src = e.target.result;
            };

            reader.readAsDataURL(file);
        }
    });
});

function redirectTumour() {
    window.location.href = "BrainTumourDetector.html";
}

function redirectStroke() {
    window.location.href = "BrainStrokeDetector.html";
}

function redirectAlzhimer() {
    window.location.href = "AlzheimerDiseaseDetector.html";
}