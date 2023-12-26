#pragma once

#include <SFML/Graphics.hpp>

#include "utils.h"

class renderWindow
{
  public:
    renderWindow::renderWindow() ;

    void Display(
      const int epoch, 
      const double loss,
      const column& gradients,
      const column& activations1,
      const column& activations2,
      const column& activations3,      
      const int gridSize,
      matrix& values);

    void ProcessEvents(bool& running);

    sf::RenderWindow window;
    sf::Font font;
    sf::Text titleStr; 
    sf::Text lossStr; 
    sf::Text gradientStr; 

    sf::RectangleShape rect;
};
