#pragma once

#include <SFML/Graphics.hpp>

#include "utils.h"

class renderWindow
{
  public:
    renderWindow::renderWindow() ;

    void Display(const int epoch, const double loss, const int gridSize, matrix& values);

    void ProcessEvents(bool& running);

    sf::RenderWindow window;
    sf::Font font;
    sf::Text titleStr; 
    sf::Text lossStr; 

    sf::RectangleShape rect;
};
