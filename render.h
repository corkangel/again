#pragma once

#include <SFML/Graphics.hpp>

class renderWindow
{
  public:
    renderWindow::renderWindow() ;

    void Display(const double loss);

    void ProcessEvents(bool& running);

    sf::RenderWindow window;
    sf::Font font;
    sf::Text titleStr; 
    sf::Text lossStr; 
};
