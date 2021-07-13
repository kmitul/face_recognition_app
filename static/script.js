function readURL(input) {
if (input.files && input.files[0]) {
    var reader = new FileReader();

    reader.onload = function(e) {
               $(input).next('.joint-img').attr('src', e.target.result);
           }

    reader.readAsDataURL(input.files[0]);
}
}

$(".joint").change(function(){
    readURL(this);
});

$('#one').css('color', 'red');
