#include "render.h"

renderWindow::renderWindow() 
    : window(sf::VideoMode(1100, 1100), "again", sf::Style::Close)
{
    font.loadFromFile("Resources/Fonts/arial.ttf");

    titleStr.setFont(font);
    titleStr.setString("Again!!!");

    lossStr.setFont(font);
    lossStr.setString("0");
    lossStr.setPosition(0,40);

    rect.setSize(sf::Vector2f(8.0f, 8.0f));
    rect.setFillColor(sf::Color::White);
}

void renderWindow::Display(const int epoch, const double loss, const int gridSize, matrix& values)
{
    window.clear(sf::Color::Black);

    window.draw(titleStr);

    char buf[200];
    snprintf(buf, sizeof(buf),  "Epoch: %d loss: %f", epoch, loss);
    lossStr.setString(buf);
    window.draw(lossStr);

    for (int y=0; y < gridSize; y++)
    {
        for (int x=0; x < gridSize; x++)
        {
            const int i = y * gridSize + x;
            const column& rgb = values[i];

            sf::Uint8 r = rgb[0] * 255;
            sf::Uint8 g = rgb[1] * 255;
            sf::Uint8 b = rgb[2] * 255;

            rect.setPosition(sf::Vector2f(20 + x*10*1.0f, 100 + y*10*1.0f));
            rect.setFillColor(sf::Color(r, g, b,  255));

            window.draw(rect);
        }
    }


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
