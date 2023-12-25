#include "render.h"

renderWindow::renderWindow() 
    : window(sf::VideoMode(1000, 600), "again", sf::Style::Close)
{
    font.loadFromFile("Resources/Fonts/arial.ttf");

    titleStr.setFont(font);
    titleStr.setString("Again!!!");

    lossStr.setFont(font);
    lossStr.setString("0");
    lossStr.setPosition(0,40);
}

void renderWindow::Display(const double loss)
{
    window.clear(sf::Color::Black);

    window.draw(titleStr);

    char buf[200];
    snprintf(buf, sizeof(buf),  "%f", loss);
    lossStr.setString(buf);

    window.draw(lossStr);

    window.display();
}

void renderWindow::ProcessEvents(bool& running)
{
    sf::Event event;
    while (window.pollEvent(event))
    {
        switch (event.type)
        {
            case sf::Event::Closed:
            {
                window.close();
                running = false;
                break;
            }
            case sf::Event::KeyReleased:
            {
                switch (event.key.code)
                {
                    case sf::Keyboard::Escape:
                    {
                        window.close();
                        running = false;
                        break;
                    }
                }
            }
        }
    }
}
