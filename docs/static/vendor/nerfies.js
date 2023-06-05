window.HELP_IMPROVE_VIDEOJS = false;

$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

  

})


$(window).on("load", function(){
    $('.author-portrait > video').each(function() {
      let el = $(this);
      let video_node = el.get(0);
      video_node.load();
    });
    $('.author-portrait > video').each(function() {
      let el = $(this);
      let video_node = el.get(0);
      el.on('canplaythrough', function(e) {
        video_node.play();
      })
    });
    $('.author-portrait').each(function() {
      $(this).mouseover(function() {
          $(this).find('.depth').css('top', '-100%');
      });
      $(this).mouseout(function() {
          $(this).find('.depth').css('top', '0%');
      });
    });


    

});

Number.prototype.clamp = function(min, max) {
  return Math.min(Math.max(this, min), max);
};


function updateHyperGrid(point) {
  const n = 20 - 1;
  let top = Math.round(n * point.y.clamp(0, 1)) * 100;
  let left = Math.round(n * point.x.clamp(0, 1)) * 100;
  $('.hyper-grid-rgb > img').css('left', -left + '%');
  $('.hyper-grid-rgb > img').css('top', -top + '%');
}